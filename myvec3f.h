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

#ifndef MY_VEC3F_H
#define MY_VEC3F_H

class MyVec3f
{
  public:
    float x, y, z;
    __device__ MyVec3f() {}
    __device__ MyVec3f(float a, float b, float c):x(a),y(b),z(c) {}
    __device__ float dot(const MyVec3f &v) const {return x*v.x+y*v.y+z*v.z;}
    __device__ float norm2() const {return x*x+y*y+z*z;}
    __device__ float norm() const {return sqrt(x*x+y*y+z*z);}
    __device__ float l1norm() const {return fabs(x)+fabs(y)+fabs(z);}
    __device__ float sum() const {return x+y+z;}

    __device__ MyVec3f operator +(const MyVec3f &v) const
    {
        MyVec3f s;
        s.x = x+v.x;
        s.y = y+v.y;
        s.z = z+v.z;
        return s;
    }

    __device__ MyVec3f operator -(const MyVec3f &v) const
    {
        MyVec3f d;
        d.x = x-v.x;
        d.y = y-v.y;
        d.z = z-v.z;
        return d;
    }

    __device__ MyVec3f operator *(float f) const
    {
        MyVec3f v;
        v.x = x*f;
        v.y = y*f;
        v.z = z*f;
        return v;
    }
    
    // Element-wise product
    __device__ MyVec3f operator *(const MyVec3f &v) const
    {
        MyVec3f s;
        s.x = x*v.x;
        s.y = y*v.y;
        s.z = z*v.z;
        return s;
    }

    // Element-wise division
    __device__ MyVec3f operator /(const MyVec3f &v) const
    {
        MyVec3f s;
        s.x = x/v.x;
        s.y = y/v.y;
        s.z = z/v.z;
        return s;
    }
};

#endif