/*
Copyright 2019 Julien Mille

This file is part of DeepFlowCUDA.

DeepFlowCUDA is free software: you can redistribute
it and/or modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

DeepFlowCUDA is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.

You should have received a copy of the GNU General Public License,
and a copy of the GNU Lesser General Public License, along with
DeepFlowCUDA. If not, see <http://www.gnu.org/licenses/>.
*/

#include "deepflowcuda.h"
#include <opencv2/cudaarithm.hpp> // for pixelwise operations on cv::cudev::GpuMat_
#include <opencv2/cudawarping.hpp> // for cv::cuda::resize
#include <iostream>

using namespace std;

// Maximum number of CUDA threads per block, per dimension, for 2D thread blocks
// It is equivalent to 32^2=1014 threads per 1D block 
#define THREADS_PER_BLOCK_2D 32

// Rounded up division, to compute number of thread blocks when launching CUDA kernels
int divUp(int a, int b)
{
    return (a + b - 1)/b;
}

DeepFlowCuda::DeepFlowCuda()
{
    // We use the same setting as DeepFlow, except for the scale factor
    fixedPointIterations = 5;
    sorIterations = 25;
    alpha = 1.0;
    beta = 32.0;
    delta = 0.1;
    gamma = 0.7;
    omega = 1.6;
    zeta = 0.1;
    epsilon = 0.01;
    scaleFactor = 0.5;
    sigma = 0; // 0.65;
    minSize = 10;

    paramsCuda = nullptr;
    copyParamsToCuda();

    current.step = 0;
    current.stepColor = 0;

    padding = 1;

    assert(createFilters());
}

DeepFlowCuda::~DeepFlowCuda()
{
    if (paramsCuda!=nullptr)
        cudaFree(paramsCuda);
}

bool DeepFlowCuda::createFilters()
{
    cv::Mat deriv, deriv5pt;
    cv::Mat o = cv::Mat::ones(1, 1, CV_32F);

    // CUDA filters for centered finite differences
    deriv.create(1, 3, CV_32F);
    deriv.at<float>(0,0) = -0.5;
    deriv.at<float>(0,1) = 0.0;
    deriv.at<float>(0,2) = 0.5;

    filtx = cv::cuda::createSeparableLinearFilter(CV_32FC3, CV_32FC3, deriv, o, cv::Point(-1, -1), cv::BORDER_REPLICATE);
    if (filtx==nullptr)
        return false;

    filty = cv::cuda::createSeparableLinearFilter(CV_32FC3, CV_32FC3, o, deriv, cv::Point(-1, -1), cv::BORDER_REPLICATE);
    if (filty==nullptr)
        return false;
    
    // CUDA filters for finite differences with 5-point stencil
    deriv5pt.create(1, 5, CV_32F);
    deriv5pt.at<float>(0,0) = 1.0/12;
    deriv5pt.at<float>(0,1) = -8.0/12;
    deriv5pt.at<float>(0,2) = 0;
    deriv5pt.at<float>(0,3) = 8.0/12;
    deriv5pt.at<float>(0,4) = -1.0/12;

    filtx5pt = cv::cuda::createSeparableLinearFilter(CV_32FC3, CV_32FC3, deriv5pt, o, cv::Point(-1, -1), cv::BORDER_REPLICATE);
    if (filtx5pt==nullptr)
        return false;
    
    filty5pt = cv::cuda::createSeparableLinearFilter(CV_32FC3, CV_32FC3, o, deriv5pt, cv::Point(-1, -1), cv::BORDER_REPLICATE);
    if (filty5pt==nullptr)
        return false;

    if (sigma!=0)
    {
        filtg = cv::cuda::createGaussianFilter(CV_32FC3, CV_32FC3, cv::Size(0,0), sigma, sigma, cv::BORDER_REPLICATE);
        if (filtg==nullptr)
            return false;
    }

    return true;
}

__global__ void warpKernel(int width, int height, int padding, int step, int stepColor, const float3 *I, const float *u, const float *v, float3 *warpedI)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i>=height || j>=width)
        return;

    int offset = (i+padding)*step + j+padding;
    MyVec3f *pWarpedI = (MyVec3f *)((float *)warpedI + ((i+padding)*stepColor + 3*(j+padding)));
    
    float x, y, xx, yy, dx, dy;
    int x1, x2, y1, y2;

    xx = j + u[offset];
    yy = i + v[offset];
    x = floor(xx);
    y = floor(yy);
    dx = xx-x;
    dy = yy-y;
            
    x1 = (int)x;
    x2 = x1+1;
    y1 = (int)y;
    y2 = y1+1;

    if (x1<0) x1=0; else if (x1>=width) x1 = width-1;
    if (x2<0) x2=0; else if (x2>=width) x2 = width-1;
    if (y1<0) y1=0; else if (y1>=height) y1 = height-1;
    if (y2<0) y2=0; else if (y2>=height) y2 = height-1;
    
    const MyVec3f *pI1 = (const MyVec3f *)((float *)I + ((y1+padding)*stepColor + 3*padding));
    const MyVec3f *pI2 = (const MyVec3f *)((float *)I + ((y2+padding)*stepColor + 3*padding));

    *pWarpedI = 
        pI1[x1]*(1.0f-dx)*(1.0f-dy) +
        pI1[x2]*dx*(1.0f-dy) +
        pI2[x1]*(1.0f-dx)*dy +
        pI2[x2]*dx*dy;
}

__global__ void averageKernel(int width, int height, int stepColor, const float3 *a, const float3 *b, float3 *c)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i>=height || j>=width)
        return;

    int offset = i*stepColor + 3*j;
    const MyVec3f *pA = (const MyVec3f *)((float *)a + offset);
    const MyVec3f *pB = (const MyVec3f *)((float *)b + offset);
    MyVec3f *pC = (MyVec3f *)((float *)c + offset);
    
    *pC = (*pA+*pB)*0.5;
}

void DeepFlowCuda::prepareBuffers(int scale)
{
    current.size = pyramid.I0[scale].size();

    current.sizePadded.width = current.size.width + 2*padding;
    current.sizePadded.height = current.size.height + 2*padding;

    current.A11.create(current.sizePadded);
    current.A12.create(current.sizePadded);
    current.A22.create(current.sizePadded);
    current.b1.create(current.sizePadded);
    current.b2.create(current.sizePadded);

    current.luminance.create(current.sizePadded);
    current.smoothX.create(current.sizePadded);
    current.smoothY.create(current.sizePadded);
    current.smoothWeight.create(current.sizePadded);

    current.smoothX.setTo(0.0);
    current.smoothY.setTo(0.0);
    current.smoothWeight.setTo(0.0);

    if (scale!=nbScales-1)
    {
        cv::cuda::resize(pyramid.ufinal[scale+1], pyramid.uinit[scale], pyramid.I0[scale].size());
        cv::cuda::resize(pyramid.vfinal[scale+1], pyramid.vinit[scale], pyramid.I0[scale].size());

        cv::cuda::multiply(pyramid.uinit[scale], 1.0/scaleFactor, pyramid.uinit[scale]);
        cv::cuda::multiply(pyramid.vinit[scale], 1.0/scaleFactor, pyramid.vinit[scale]);
    }

    if (padding!=0)
    {
        cv::cuda::copyMakeBorder(pyramid.I0[scale], current.I0, padding, padding, padding, padding, cv::BORDER_REPLICATE);
        cv::cuda::copyMakeBorder(pyramid.uinit[scale], current.u, padding, padding, padding, padding, cv::BORDER_REPLICATE);
        cv::cuda::copyMakeBorder(pyramid.vinit[scale], current.v, padding, padding, padding, padding, cv::BORDER_REPLICATE);

        if (beta!=0)
        {
            cv::cuda::copyMakeBorder(pyramid.udesc[scale], current.udesc, padding, padding, padding, padding, cv::BORDER_REPLICATE);
            cv::cuda::copyMakeBorder(pyramid.vdesc[scale], current.vdesc, padding, padding, padding, padding, cv::BORDER_REPLICATE);
            cv::cuda::copyMakeBorder(pyramid.descWeight[scale], current.descWeight, padding, padding, padding, padding, cv::BORDER_REPLICATE);
        }
    }
    else {
        pyramid.I0[scale].copyTo(current.I0);
        pyramid.uinit[scale].copyTo(current.u);
        pyramid.vinit[scale].copyTo(current.v);

        if (beta!=0)
        {
            pyramid.udesc[scale].copyTo(current.udesc);
            pyramid.vdesc[scale].copyTo(current.vdesc);
            pyramid.descWeight[scale].copyTo(current.descWeight);
        }
    }

    current.step = current.A11.step/sizeof(float);
    current.stepColor = current.I0.step/sizeof(float);

    // Computing an average of the current and warped next frames (to compute the derivatives on) and temporal derivative Iz
    dim3 threadsPerBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D, 1), blocksPerGrid;

    // Otherwise, pyramid.uinit[scale] and pyramid.vinit[scale] are already initialized
    cv::cudev::GpuMat_<float3> gpuwarpedI, gpuaveragedI;

    // Average everywhere
    gpuwarpedI.create(current.size);

    blocksPerGrid = dim3(divUp(current.size.width, threadsPerBlock.x), divUp(current.size.height, threadsPerBlock.y), 1);
    warpKernel<<<blocksPerGrid, threadsPerBlock>>>(current.size.width, current.size.height, 0, pyramid.uinit[scale].step/sizeof(float), pyramid.I1[scale].step/sizeof(float),
        pyramid.I1[scale][0], pyramid.uinit[scale][0], pyramid.vinit[scale][0], gpuwarpedI[0]);

    if (padding!=0)
        cv::cuda::copyMakeBorder(gpuwarpedI, gpuwarpedI, padding, padding, padding, padding, cv::BORDER_REPLICATE);

    // Average everywhere
    gpuaveragedI.create(current.sizePadded);
    
    blocksPerGrid = dim3(divUp(current.sizePadded.width, threadsPerBlock.x), divUp(current.size.height, threadsPerBlock.y), 1);
    averageKernel<<<blocksPerGrid, threadsPerBlock>>>(current.sizePadded.width, current.sizePadded.height, current.stepColor, current.I0[0], gpuwarpedI[0], gpuaveragedI[0]);
    cv::cuda::subtract(gpuwarpedI, current.I0, current.Iz);

    filtx5pt->apply(gpuaveragedI, current.Ix);
    filty5pt->apply(gpuaveragedI, current.Iy);

    filtx5pt->apply(current.Iz, current.Ixz);
    filty5pt->apply(current.Iz, current.Iyz);

    filtx5pt->apply(current.Ix, current.Ixx);
    filty5pt->apply(current.Ix, current.Ixy);
    filty5pt->apply(current.Iy, current.Iyy);

    current.u.copyTo(current.utmp);
    current.v.copyTo(current.vtmp);

    current.du.create(current.sizePadded);
    current.dv.create(current.sizePadded);
    current.du.setTo(0.0);
    current.dv.setTo(0.0);
}

__global__ void structureTensorKernel(int width, int height, int step, int stepColor, const float3 *Ix, const float3 *Iy, float *stx2, float *stxy, float *sty2)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i>=height || j>=width)
        return;

    int offset = i*step + j;
    int offsetColor = i*stepColor + 3*j;
    const MyVec3f *pIx = (const MyVec3f *)((float *)Ix + offsetColor);
    const MyVec3f *pIy = (const MyVec3f *)((float *)Iy + offsetColor);
    
    stx2[offset] = (*pIx).dot(*pIx);
    stxy[offset] = (*pIx).dot(*pIy);
    sty2[offset] = (*pIy).dot(*pIy);
}

__global__ void minEigenvalueKernel(int width, int height, int step, const float *stx2, const float *stxy, const float *sty2, float *minEigen)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i>=height || j>=width)
        return;

    int offset = i*step + j;
    const float *pstx2 = stx2 + offset;
    const float *pstxy = stxy + offset;
    const float *psty2 = sty2 + offset;

    float t = 0.5f * (*pstx2 + *psty2);
    float t2 = t*t + (*pstxy)*(*pstxy) - (*pstx2)*(*psty2);
    minEigen[offset] = t - (t2<=0.0f?0.0f:sqrtf(t2)); // may be negative due to floating points approximation
}

__global__ void matchingScoreKernel(int width, int height, int step, int stepColor,
    const float3 *I0, const float3 *I1,
    const float3 *I0x, const float3 *I0y, const float3 *I1x, const float3 *I1y, 
    const float *udesc, const float *vdesc, const float *minEigen, float *descWeight)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i>=height || j>=width)
        return;

    int offset = i*step + j;
    int offsetColor = i*stepColor + 3*j;    
    const MyVec3f *pI0 = (const MyVec3f *)((float *)I0 + offsetColor);
    const MyVec3f *pI0x = (const MyVec3f *)((float *)I0x + offsetColor);
    const MyVec3f *pI0y = (const MyVec3f *)((float *)I0y + offsetColor);
    
    float xw, yw;
    xw = (int)(j + udesc[offset]);
    yw = (int)(i + vdesc[offset]);
    if (xw<0) xw=0;
    else if (xw>width-1) xw = width-1;
    if (yw<0) yw=0;
    else if (yw>height-1) yw = height-1;

    int offsetColor2 = yw*stepColor + 3*xw;
    const MyVec3f *pI1 = (const MyVec3f *)((float *)I1 + offsetColor2);
    const MyVec3f *pI1x = (const MyVec3f *)((float *)I1x + offsetColor2);
    const MyVec3f *pI1y = (const MyVec3f *)((float *)I1y + offsetColor2);

    float gradWeight = 1.0;
    float flow_sigma_score = 50.0f;
    float mul_coef = 10.0f;

    float flowscore = (*pI0 - *pI1).l1norm() + gradWeight * ((*pI0x - *pI1x).l1norm() + (*pI0y - *pI1y).l1norm());
    float t2 = minEigen[offset];

    float t = 1.0f/(flow_sigma_score*sqrtf(2.0f*M_PI));
    float sigmascore2 = -0.5f/(flow_sigma_score*flow_sigma_score);

    t2 = t2<=0.0?0.0:sqrtf(t2);
    descWeight[offset] = mul_coef * t2 * t * expf( flowscore*flowscore*sigmascore2 );
    if (descWeight[offset]<0.0)  // may be negative due to floating points approximation
        descWeight[offset] = 0.0;
}

void DeepFlowCuda::computeDescWeight()
{
    // Input : pyramid.I0[0], pyramid.I1[0]
    // Output : pyramid.descWeight[0]
    cv::cudev::GpuMat_<float3> I0x, I0y;

    // Structure tensor
    cv::cudev::GpuMat_<float> stx2, stxy, sty2, minEigen;

    filtx->apply(pyramid.I0[0], I0x);
    filty->apply(pyramid.I0[0], I0y);

    stx2.create(I0x.size());
    stxy.create(I0x.size());
    sty2.create(I0x.size());

    dim3 threadsPerBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D, 1), blocksPerGrid(divUp(I0x.cols, threadsPerBlock.x), divUp(I0x.rows, threadsPerBlock.y), 1);
    
    // No padding here
    structureTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(I0x.cols, I0x.rows, stx2.step/sizeof(float), I0x.step/sizeof(float),
        I0x[0], I0y[0], stx2[0], stxy[0], sty2[0]);

    // Smooth structure tensor
    shared_ptr<cv::cuda::Filter> fg = cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(0,0), 3.0, 3.0, cv::BORDER_REPLICATE);
    fg->apply(stx2, stx2);
    fg->apply(stxy, stxy);
    fg->apply(sty2, sty2);

    minEigen.create(I0x.size());
    pyramid.descWeight[0].create(I0x.size());

    minEigenvalueKernel<<<blocksPerGrid, threadsPerBlock>>>(I0x.cols, I0x.rows, stx2.step/sizeof(float), stx2[0], stxy[0], sty2[0], minEigen[0]);
    
    cv::cudev::GpuMat_<float3> I0x5pt, I0y5pt, I1x5pt, I1y5pt;

    filtx5pt->apply(pyramid.I0[0], I0x5pt);
    filty5pt->apply(pyramid.I0[0], I0y5pt);
    filtx5pt->apply(pyramid.I1[0], I1x5pt);
    filty5pt->apply(pyramid.I1[0], I1y5pt);

    matchingScoreKernel<<<blocksPerGrid, threadsPerBlock>>>(I0x.cols, I0x.rows, stx2.step/sizeof(float), I0x.step/sizeof(float),
        pyramid.I0[0][0], pyramid.I1[0][0],
        I0x5pt[0], I0y5pt[0], I1x5pt[0], I1y5pt[0],
        pyramid.udesc[0][0], pyramid.vdesc[0][0], minEigen[0], pyramid.descWeight[0][0]);
}

// Computed on all pixels (including padding)
__global__ void luminanceKernel(int width, int height, int step, int stepColor, const float3 *I, float *lum)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i>=height || j>=width)
        return;

    int offsetI = i*stepColor + 3*j;
    int offset = i*step+j;

    const MyVec3f *pI = (const MyVec3f *)((const float *)I + offsetI);
    lum[offset] = 0.299f*pI->z + 0.587f*pI->y + 0.114f*pI->x;
}

__global__ void smoothnessWeightKernel(int width, int height, int padding, int step, const float *lum, float *smoothWeight, float coef)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i>=height || j>=width)
        return;

    int offset = (i+padding)*step + padding + j;
    float lumx = (lum[offset+1]-lum[offset-1])*0.5;
    float lumy = (lum[offset+step]-lum[offset-step])*0.5;
    smoothWeight[offset] = 0.5f*expf(-coef*sqrtf(lumx*lumx+lumy*lumy));
}

void DeepFlowCuda::computeSmoothnessWeight()
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D, 1), blocksPerGrid;
    
    // Luminance everywhere (including padding)
    blocksPerGrid = dim3(divUp(current.sizePadded.width, threadsPerBlock.x), divUp(current.sizePadded.height, threadsPerBlock.y), 1);
    luminanceKernel<<<blocksPerGrid, threadsPerBlock>>>(current.sizePadded.width, current.sizePadded.height, current.step, current.stepColor, current.I0[0], current.luminance[0]);

    blocksPerGrid = dim3(divUp(current.size.width, threadsPerBlock.x), divUp(current.size.height, threadsPerBlock.y), 1);
    smoothnessWeightKernel<<<blocksPerGrid, threadsPerBlock>>>(current.size.width, current.size.height, padding, current.step, current.luminance[0], current.smoothWeight[0], 5.0/255.0);
}

__global__ void dataTermKernel(
    const float *params, int width, int height, int padding, int step, int stepColor,
    const float3 *Ix, const float3 *Iy, const float3 *Iz,
    const float3 *Ixx, const float3 *Ixy, const float3 *Iyy,
    const float3 *Ixz, const float3 *Iyz,
    float *A11, float *A12, float *A22, float *b1, float *b2,
    const float *du, const float *dv)
{
    // float params[7] = {alpha, beta, delta, gamma, omega, zeta, epsilon};
    float delta = params[2], gamma = params[3], zeta = params[5], epsilon = params[6];

    float zeta_squared = zeta * zeta;
    float epsilon_squared = epsilon * epsilon;
    
    const MyVec3f *pIx, *pIy, *pIz;
    const MyVec3f *pIxx, *pIxy, *pIyy, *pIxz, *pIyz;
    const float *pdU, *pdV;
    float *pa11, *pa12, *pa22, *pb1, *pb2;

    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i>=height || j>=width)
        return;

    // stepColor is in number of float elements!
    int offsetColor = (i+padding)*stepColor + 3*(padding+j);
    pIx = (const MyVec3f *)((float *)Ix + offsetColor);
    pIy = (const MyVec3f *)((float *)Iy + offsetColor);
    pIz = (const MyVec3f *)((float *)Iz + offsetColor);
    pIxx = (const MyVec3f *)((float *)Ixx + offsetColor);
    pIxy = (const MyVec3f *)((float *)Ixy + offsetColor);
    pIyy = (const MyVec3f *)((float *)Iyy + offsetColor);
    pIxz = (const MyVec3f *)((float *)Ixz + offsetColor);
    pIyz = (const MyVec3f *)((float *)Iyz + offsetColor);
    
    int offset = (i+padding)*step + padding+j;
    pa11 = A11 + offset;
    pa12 = A12 + offset;
    pa22 = A22 + offset;
    pb1 = b1 + offset;
    pb2 = b2 + offset;
    pdU = du + offset;
    pdV = dv + offset;

    *pa11 = 0;
    *pa12 = 0;
    *pa22 = 0;
    *pb1 = 0;
    *pb2 = 0;

    // Color constancy
    if (delta!=0.0)
    {
        MyVec3f dnorm(zeta_squared, zeta_squared, zeta_squared);
        float hdover3 = delta*0.5f/3.0f;
        float mask = 1.0;

        MyVec3f ngradI = *pIx*(*pIx) + *pIy*(*pIy) + dnorm;

        MyVec3f Ik1z = *pIz + *pIx*(*pdU) + *pIy*(*pdV);
        float tmp = mask*hdover3/sqrt((Ik1z*Ik1z/ngradI).sum()+epsilon_squared);
        MyVec3f ti = MyVec3f(tmp, tmp, tmp)/ngradI;

        *pa11 += (ti*(*pIx)*(*pIx)).sum();
        *pa12 += (ti*(*pIx)*(*pIy)).sum();
        *pa22 += (ti*(*pIy)*(*pIy)).sum();
        *pb1 -= (ti*(*pIx)*pIz[j]).sum();
        *pb2 -= (ti*(*pIy)*pIz[j]).sum();
    }

    // Gradient constancy
    if (gamma!=0)
    {
        MyVec3f dnorm(zeta_squared, zeta_squared, zeta_squared);
        float hgover3 = gamma*0.5f/3.0f;
        float mask = 1.0;

        MyVec3f nx = *pIxx*(*pIxx) + *pIxy*(*pIxy) + dnorm;
        MyVec3f ny = *pIyy*(*pIyy) + *pIxy*(*pIxy) + dnorm;

        MyVec3f tmpx = *pIxz + *pIxx*(*pdU) + *pIxy*(*pdV);
        MyVec3f tmpy = *pIyz + *pIxy*(*pdU) + *pIyy*(*pdV);
    
        float tmp = mask*hgover3/sqrt((tmpx*tmpx/nx).sum() + (tmpy*tmpy/ny).sum() + epsilon_squared);

        MyVec3f tix = MyVec3f(tmp, tmp, tmp)/nx;
        MyVec3f tiy = MyVec3f(tmp, tmp, tmp)/ny;

        *pa11 += (tix*(*pIxx)*(*pIxx) + tiy*(*pIxy)*(*pIxy)).sum();
        *pa12 += (tix*(*pIxx)*(*pIxy) + tiy*(*pIxy)*(*pIyy)).sum();
        *pa22 += (tix*(*pIxy)*(*pIxy) + tiy*(*pIyy)*(*pIyy)).sum();
        
        *pb1 -= (tix*(*pIxx)*(*pIxz) + tiy*(*pIxy)*(*pIyz)).sum();
        *pb2 -= (tix*(*pIxy)*(*pIxz) + tiy*(*pIyy)*(*pIyz)).sum();
    }
}

void DeepFlowCuda::computeDataTerm()
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D, 1), blocksPerGrid(divUp(current.size.width, threadsPerBlock.x), divUp(current.size.height, threadsPerBlock.y), 1);
    
    dataTermKernel<<<blocksPerGrid, threadsPerBlock>>>(
        paramsCuda, current.size.width, current.size.height, padding, current.step, current.stepColor,
        current.Ix[0], current.Iy[0], current.Iz[0],
        current.Ixx[0], current.Ixy[0], current.Iyy[0], current.Ixz[0], current.Iyz[0],
        current.A11[0], current.A12[0], current.A22[0], current.b1[0], current.b2[0],
        current.du[0], current.dv[0]);
}

__global__ void matchingTermKernel(
    const float *params, int width, int height, int padding, int step,
    float *A11, float *A22, float *b1, float *b2,
    const float *u, const float *v, const float *utmp, const float *vtmp, const float *udesc, const float *vdesc, const float *descWeight)
{
    // float params[7] = {alpha, beta, delta, gamma, omega, zeta, epsilon};
    float beta = params[1], epsilon = params[6];

    float epsilon_squared = epsilon*epsilon;

    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i>=height || j>=width)
        return;
    
    int offset = (i+padding)*step + padding+j;
    
    const float *pudesc = udesc + offset;
    const float *pvdesc = vdesc + offset;

    float tmpx = utmp[offset] - *pudesc;
    float tmpy = vtmp[offset] - *pvdesc;
    float tmp = 0.5*descWeight[offset]*beta/sqrt(tmpx*tmpx+tmpy*tmpy+epsilon_squared);
    A11[offset] += tmp;
    A22[offset] += tmp;
    b1[offset] -= tmp*(u[offset] - *pudesc);
    b2[offset] -= tmp*(v[offset] - *pvdesc);
}

void DeepFlowCuda::computeMatchingTerm()
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D, 1), blocksPerGrid(divUp(current.size.width, threadsPerBlock.x), divUp(current.size.height, threadsPerBlock.y), 1);
    
    matchingTermKernel<<<blocksPerGrid, threadsPerBlock>>>(
        paramsCuda, current.size.width, current.size.height, padding, current.step,
        current.A11[0], current.A22[0], current.b1[0], current.b2[0],
        current.u[0], current.v[0], current.utmp[0], current.vtmp[0], current.udesc[0], current.vdesc[0], current.descWeight[0]);
}

__global__ void smoothnessTermKernel(
    const float *params, int width, int height, int padding, int step,
    float *smoothX, float *smoothY, const float *smoothWeight,
    const float *utmp, const float *vtmp)
{
    // float params[7] = {alpha, beta, delta, gamma, omega, zeta, epsilon};
    float alpha = params[0], epsilon = params[6]; 

    float epsilon_smooth = epsilon*epsilon; // 0.001f*0.001f;
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i>=height || j>=width)
        return;

    int offset = (i+padding)*step + padding + j;

    float *psmoothX = smoothX + offset;
    float *psmoothY = smoothY + offset;
    const float *psmoothWeight = smoothWeight + offset; 
    const float *pu = utmp + offset;
    const float *pv = vtmp + offset;

    float ux1 = pu[1]-pu[0];
    float vx1 = pv[1]-pv[0];
    float uy1 = pu[step]-pu[0];
    float vy1 = pv[step]-pv[0];

    float ux2 = (pu[1]-pu[-1])*0.5;
    float vx2 = (pv[1]-pv[-1])*0.5;
    float uy2 = (pu[step]-pu[-step])*0.5;
    float vy2 = (pv[step]-pv[-step])*0.5;
    
    float tmpu = 0.5*(uy2 + (pu[step+1]-pu[-step+1])*0.5);
    float uxsq = ux1*ux1 + tmpu*tmpu;
    float tmpv = 0.5*(vy2 + (pv[step+1]-pv[-step+1])*0.5);
    float vxsq = vx1*vx1 + tmpv*tmpv;

    *psmoothX = alpha*0.5*(psmoothWeight[0]+psmoothWeight[1])/sqrt(uxsq+vxsq+epsilon_smooth);

    tmpu = 0.5*(ux2 + (pu[step+1]-pu[step-1])*0.5);
    float uysq = uy1*uy1 + tmpu*tmpu;
    tmpv = 0.5*(vx2 + (pv[step+1]-pv[step-1])*0.5);
    float vysq = vy1*vy1 + tmpv*tmpv;

    *psmoothY = alpha*0.5*(psmoothWeight[0]+psmoothWeight[step])/sqrt(uysq+vysq+epsilon_smooth);
}

__global__ void applySmoothKernel(
    int width, int height, int padding, int step,
    float *b1, float *b2,
    const float *smoothX, const float *smoothY,
    const float *u, const float *v
    )
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i>=height || j>=width)
        return;

    int offset = (i+padding)*step + padding + j;
    const float *pu = u + offset;
    const float *pv = v + offset;
    const float *psx = smoothX + offset;
    const float *psy = smoothY + offset;
    
    // V1
    b1[offset] += -psx[-1]*(pu[0]-pu[-1]) + psx[0]*(pu[1]-pu[0]) -psy[-step]*(pu[0]-pu[-step]) + psy[0]*(pu[step]-pu[0]);
    b2[offset] += -psx[-1]*(pv[0]-pv[-1]) + psx[0]*(pv[1]-pv[0]) -psy[-step]*(pv[0]-pv[-step]) + psy[0]*(pv[step]-pv[0]);

    // V2
    // b1[offset] += smoothX[offset]*(pu[1]-2*pu[0]+pu[-1]) + smoothY[offset]*(pu[step]-2*pu[0]+pu[-step]);
    // b2[offset] += smoothX[offset]*(pv[1]-2*pv[0]+pv[-1]) + smoothY[offset]*(pv[step]-2*pv[0]+pv[-step]);    
}

void DeepFlowCuda::computeSmoothnessTerm()
{
   dim3 threadsPerBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D, 1), blocksPerGrid(divUp(current.size.width, threadsPerBlock.x), divUp(current.size.height, threadsPerBlock.y), 1);
    
   smoothnessTermKernel<<<blocksPerGrid, threadsPerBlock>>>(
        paramsCuda, current.size.width, current.size.height, padding, current.step,
        current.smoothX[0], current.smoothY[0], current.smoothWeight[0],
        current.utmp[0], current.vtmp[0]);

    applySmoothKernel<<<blocksPerGrid, threadsPerBlock>>>(
        current.size.width, current.size.height, padding, current.step,
        current.b1[0], current.b2[0], current.smoothX[0], current.smoothY[0], current.u[0], current.v[0]);
}

__global__ void RedBlackSORKernel(
    const float *params, int width, int height, int padding, int step, bool redpass,
    const float *a11, const float *a12, const float *a22, const float *b1, const float *b2,
    const float *smoothX, const float *smoothY,
    float *du, float *dv)
{
    // float params[7] = {alpha, beta, delta, gamma, omega, zeta, epsilon};
    float omega = params[4];

    int halfWidth = width/2 + width%2;
    int widthRow; // Width of current row
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i>=height)
        return;

    if (width%2==0)
        widthRow = halfWidth;
    else {
        if (i%2==0)
        {
            if (redpass) widthRow = halfWidth;
            else widthRow = halfWidth-1;
        }
        else {
            if (redpass) widthRow = halfWidth-1;
            else widthRow = halfWidth;
        }
    }
    if (j>=widthRow)
        return;

    int offset = (i+padding)*step + padding + j*2;
    if ((redpass && i%2==1) || (!redpass && i%2==0))
        offset++;

    float sigma_u,sigma_v,sum_dpsis,A11,A22,A12,B1,B2,det;
    
    sigma_u = 0.0f;
    sigma_v = 0.0f;
    sum_dpsis = 0.0f;
    
    if (i>0) {
        sigma_u -= smoothY[offset-step] * du[offset-step];
        sigma_v -= smoothY[offset-step] * dv[offset-step];
        sum_dpsis += smoothY[offset-step];
    }

    if(j>0){
        sigma_u -= smoothX[offset-1] * du[offset-1];
        sigma_v -= smoothX[offset-1] * dv[offset-1];
        sum_dpsis += smoothX[offset-1];
    }
    if(i<height-1){
        sigma_u -= smoothY[offset] * du[offset+step];
        sigma_v -= smoothY[offset] * dv[offset+step];
        sum_dpsis += smoothY[offset];
    }
    if(j<halfWidth-1){
        sigma_u -= smoothX[offset] * du[offset+1];
        sigma_v -= smoothX[offset] * dv[offset+1];
        sum_dpsis += smoothX[offset];
    }

    A11 = a11[offset] + sum_dpsis;
    A12 = a12[offset];
    A22 = a22[offset] + sum_dpsis;
    det = A11*A22-A12*A12;
    B1 = b1[offset]-sigma_u;
    B2 = b2[offset]-sigma_v;

    du[offset] = (1.0f-omega)*du[offset] + omega*( A22*B1-A12*B2)/det;
    dv[offset] = (1.0f-omega)*dv[offset] + omega*(-A12*B1+A11*B2)/det;
}

void DeepFlowCuda::RedBlackSOR()
{
    int halfWidth = current.size.width/2 + current.size.width%2;

   dim3 threadsPerBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D, 1), blocksPerGrid(divUp(halfWidth, threadsPerBlock.x), divUp(current.size.height, threadsPerBlock.y), 1);
    
    for (int iter = 0; iter < sorIterations; iter++)
    {
        RedBlackSORKernel<<<blocksPerGrid, threadsPerBlock>>>(
            paramsCuda, current.size.width, current.size.height, padding, current.step, true,
            current.A11[0], current.A12[0], current.A22[0], current.b1[0], current.b2[0],
            current.smoothX[0], current.smoothY[0],
            current.du[0], current.dv[0]);

        RedBlackSORKernel<<<blocksPerGrid, threadsPerBlock>>>(
            paramsCuda, current.size.width, current.size.height, padding, current.step, false,
            current.A11[0], current.A12[0], current.A22[0], current.b1[0], current.b2[0],
            current.smoothX[0], current.smoothY[0],
            current.du[0], current.dv[0]);
    }
}

void DeepFlowCuda::computeFlow(const cv::Mat &I0, const cv::Mat &I1, cv::Mat &flow)
{
    assert(!I0.empty() && I0.type()==CV_32FC3);
    assert(I1.size()==I0.size() && I1.type()==CV_32FC3);
    
    if (flow.empty() || flow.size()!=I0.size() || flow.type()!=CV_32FC2)
    {
        if (beta!=0)
        {
            cout<<"No correct input flow was provided but weight of matching term is non-zero. Setting it to zero..."<<endl;
            beta = 0;
            copyParamsToCuda();
        }

        flow.create(I0.size(), CV_32FC2);
        flow.setTo(0);
    }
    cv::Mat uv[2];
    
    cv::split(flow, uv);
   
    // Build pyramid
    int widthTmp =  I0.cols, heightTmp = I1.rows;
    nbScales = 0;
    while (widthTmp>=minSize && heightTmp>=minSize)
    {
        widthTmp*=scaleFactor;
        heightTmp*=scaleFactor;
        nbScales++;
    }
    if (nbScales==0)
        nbScales = 1;

    cout<<"Nb scales = "<<nbScales<<endl;

    cv::Mat rgba;

    pyramid.I0.resize(nbScales);
    pyramid.I1.resize(nbScales);
    pyramid.uinit.resize(nbScales);
    pyramid.vinit.resize(nbScales);
    pyramid.ufinal.resize(nbScales);
    pyramid.vfinal.resize(nbScales);
    pyramid.udesc.resize(nbScales);
    pyramid.vdesc.resize(nbScales);
    pyramid.descWeight.resize(nbScales);
    
    pyramid.scale.resize(nbScales);

    // cv::cvtColor(I0, rgba, cv::COLOR_BGR2BGRA);
    // cv::GaussianBlur(I0, rgba, cv::Size(0,0), sigma);
    pyramid.I0[0].upload(I0);

    // cv::cvtColor(I1, rgba, cv::COLOR_BGR2BGRA);
    // cv::GaussianBlur(I1, rgba, cv::Size(0,0), sigma);
    pyramid.I1[0].upload(I1);

    if (sigma!=0)
    {
        filtg->apply(pyramid.I0[0], pyramid.I0[0]);
        filtg->apply(pyramid.I1[0], pyramid.I1[0]);
    }

    pyramid.scale[0] = 1.0;

    if (beta!=0)
    {
        pyramid.udesc[0].upload(uv[0]);
        pyramid.vdesc[0].upload(uv[1]);
        computeDescWeight(); // Computes pyramid.descWeight[0]
    }

    cout<<"Scale 0:"<<pyramid.I0[0].size()<<endl;
    for (int k=1; k<nbScales; k++)
    {
        cv::cuda::resize(pyramid.I0[k-1], pyramid.I0[k], cv::Size(0,0), scaleFactor, scaleFactor);
        cv::cuda::resize(pyramid.I1[k-1], pyramid.I1[k], cv::Size(0,0), scaleFactor, scaleFactor);

        if (beta!=0)
        {
            cv::cuda::resize(pyramid.udesc[k-1], pyramid.udesc[k], cv::Size(0,0), scaleFactor, scaleFactor);
            cv::cuda::resize(pyramid.vdesc[k-1], pyramid.vdesc[k], cv::Size(0,0), scaleFactor, scaleFactor);
            cv::cuda::resize(pyramid.descWeight[k-1], pyramid.descWeight[k], cv::Size(0,0), scaleFactor, scaleFactor);
                    
            cv::cuda::multiply(pyramid.udesc[k], scaleFactor, pyramid.udesc[k]);
            cv::cuda::multiply(pyramid.vdesc[k], scaleFactor, pyramid.vdesc[k]);
            cv::cuda::multiply(pyramid.descWeight[k], 1.0/(scaleFactor*scaleFactor), pyramid.descWeight[k]);
        }
        cout<<"Scale "<<k<<":"<<pyramid.I0[k].size()<<endl;
    
        pyramid.scale[k] = pyramid.scale[k-1]*scaleFactor;
    }

    // Set the initial flow at the largest scale
    if (beta!=0)
    {
        pyramid.udesc[nbScales-1].copyTo(pyramid.uinit[nbScales-1]);
        pyramid.vdesc[nbScales-1].copyTo(pyramid.vinit[nbScales-1]);
    }
    else {
        cv::cudev::GpuMat_<float> uinitTmp, vinitTmp;
        uinitTmp.upload(uv[0]);
        vinitTmp.upload(uv[1]);

        cv::cuda::resize(uinitTmp, pyramid.uinit[nbScales-1], pyramid.I0[nbScales-1].size());
        cv::cuda::resize(vinitTmp, pyramid.vinit[nbScales-1], pyramid.I0[nbScales-1].size());
        cv::cuda::multiply(pyramid.uinit[nbScales-1], pyramid.scale[nbScales-1], pyramid.uinit[nbScales-1]);
        cv::cuda::multiply(pyramid.vinit[nbScales-1], pyramid.scale[nbScales-1], pyramid.vinit[nbScales-1]);
    }

    for (int k=nbScales-1; k>=0; k--)
        computeOneLevel(k);

    // At the end, we have ufinal and vfinal in pyramid.ufinal[0]
    pyramid.ufinal[0].download(uv[0]);
    pyramid.vfinal[0].download(uv[1]);

    cv::merge(uv, 2, flow);
}

void DeepFlowCuda::computeOneLevel(int scale)
{
    prepareBuffers(scale);

    float betaSave = beta;
    float bk = 0.45;

    if (bk>0.0f && nbScales>1)
    {
        beta = betaSave * pow((float)scale/(float)(nbScales-1), bk);
        copyParamsToCuda();
    }

    computeSmoothnessWeight();

    for (int i = 0; i < fixedPointIterations; i++)
    {
        computeDataTerm(); // Initializes A11, A12, A22, b1 and b2
        if (beta!=0)
            computeMatchingTerm();

        if (alpha!=0)
            computeSmoothnessTerm(); // Updates b1 and b2
        
        RedBlackSOR();

        current.utmp = current.u + current.du;
        current.vtmp = current.v + current.dv;
    }

    beta = betaSave;

    if (padding!=0)
    {
        int width = pyramid.I0[scale].cols;
        int height = pyramid.I0[scale].rows;
        
        current.utmp(cv::Rect(padding, padding, width, height)).copyTo(pyramid.ufinal[scale]);
        current.vtmp(cv::Rect(padding, padding, width, height)).copyTo(pyramid.vfinal[scale]);
    }
    else {
        current.utmp.copyTo(pyramid.ufinal[scale]);
        current.vtmp.copyTo(pyramid.vfinal[scale]);
    }
}

cv::Mat DeepFlowCuda::toCPU(const cv::cudev::GpuMat_<float> &m) const
{
    cv::Mat a;
    m.download(a);
    return a(cv::Rect(padding, padding, m.cols-2*padding, m.rows-2*padding)).clone();
}

cv::Mat DeepFlowCuda::toCPU(const cv::cudev::GpuMat_<float3> &m) const
{
    cv::Mat a;
    m.download(a);
    cv::cvtColor(a, a, cv::COLOR_BGRA2BGR);
    return a(cv::Rect(padding, padding, m.cols-2*padding, m.rows-2*padding)).clone();
}

void DeepFlowCuda::copyParamsToCuda()
{
    int nbParams = 7;
    if (paramsCuda==nullptr && cudaMalloc(&paramsCuda, nbParams*sizeof(float))!=cudaSuccess)
    {
        paramsCuda = nullptr;
        cout<<"cudaMalloc error"<<endl;
        return;
    }

    float params[] = {alpha, beta, delta, gamma, omega, zeta, epsilon};
    cudaMemcpy(paramsCuda, params, nbParams*sizeof(float), cudaMemcpyHostToDevice);
}