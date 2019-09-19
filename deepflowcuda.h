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

#ifndef DEEP_FLOW_CUDA_H
#define DEEP_FLOW_CUDA_H

#include <opencv2/core.hpp>
#include <opencv2/cudev.hpp> // For template cv::cudev::GpuMat_ -> needs to be compiled with nvcc !
#include <opencv2/cudafilters.hpp> // For cv::cuda::Filter
#include <vector>

#include "myvec3f.h"

class DeepFlowCuda
{
  // Member variables
  protected:
    int fixedPointIterations, sorIterations;
    float omega; // Update parameter in SOR iterations
    float alpha; // Weight of smoothing term
    float beta; // Weight of matching term
    float delta; // Weight of color constancy in data term
    float gamma; // Weight of gradient constancy in data term
    float zeta; // Regularization parameter (added to norms in data term)
    float epsilon; // Regularization parameter in Psi function
    float scaleFactor; // Scale factor between two succesive levels
    float sigma; // Standard deviation of presmoothing Gaussian filter
    int nbScales; // Number of scales (levels). Will be calculated from minSize, scaleFactor and size of input image
    int padding;
    int minSize; // Minimum width or height at the highest level (at the coarsest scale)
    float *paramsCuda;

    // Padded images and flows at current scale
    struct {
        cv::cudev::GpuMat_<float3> I0, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz;
        cv::cudev::GpuMat_<float> A11, A12, A22, b1, b2;
        cv::cudev::GpuMat_<float> smoothX, smoothY, luminance, smoothWeight;

        cv::cudev::GpuMat_<float> utmp, vtmp; // flow that is updated in each fixed point iteration
        cv::cudev::GpuMat_<float> du, dv;     // flow increment, updated in each SOR iteration
        cv::cudev::GpuMat_<float> u, v;       // red-black-buffer version of the input flow
        cv::cudev::GpuMat_<float> udesc, vdesc, descWeight; // Descriptor used in matching term

        int step;
        int stepColor;
    
        cv::Size size, sizePadded;
    } current;

    // Non-padded images and flows at all scales
    struct {
        std::vector<cv::cudev::GpuMat_<float3> > I0, I1;
        std::vector<cv::cudev::GpuMat_<float> > uinit, vinit, ufinal, vfinal;
        std::vector<cv::cudev::GpuMat_<float> > udesc, vdesc, descWeight;
        std::vector<float> scale;
    } pyramid;
    
    // Derivative filters, and Gaussian filter
    std::shared_ptr<cv::cuda::Filter> filtg, filtx, filty, filtx5pt, filty5pt;

  // Member functions
  public:
    DeepFlowCuda();
    ~DeepFlowCuda();

    void computeFlow(const cv::Mat &I0, const cv::Mat &I1, cv::Mat &flow);
    
    int getFixedPointIterations() const { return fixedPointIterations; }
    void setFixedPointIterations(int val) { fixedPointIterations = val; }
    int getSorIterations() const { return sorIterations; }
    void setSorIterations(int val) { sorIterations = val; }
    float getOmega() const { return omega; }
    void setOmega(float val) { omega = val; }
    float getAlpha() const { return alpha; }
    void setAlpha(float val) { alpha = val; }
    float getBeta() const { return beta; }
    void setBeta(float val) { beta = val; }
    float getDelta() const { return delta; }
    void setDelta(float val) { delta = val; }
    float getGamma() const { return gamma; }
    void setGamma(float val) { gamma = val; }

  protected:
    bool createFilters();

    void prepareBuffers(int scale);

    void computeDescWeight();

    void computeDataTerm();
    void computeMatchingTerm();

    void computeSmoothnessWeight();
    void computeSmoothnessTerm();
    
    void RedBlackSOR();
    
    void computeOneLevel(int);

    // Remove padding and move to CPU
    cv::Mat toCPU(const cv::cudev::GpuMat_<float> &) const ;
    cv::Mat toCPU(const cv::cudev::GpuMat_<float3> &) const;

    void copyParamsToCuda();
};

#endif
