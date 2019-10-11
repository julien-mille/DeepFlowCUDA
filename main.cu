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

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/imgcodecs.hpp> // For cv::imread
#include <opencv2/video.hpp> // For cv::readOpticalFlow, cv::writeOpticalFlow
#include "deepflowcuda.h"

using namespace std;

float L2Distance(const cv::Mat &flow1, const cv::Mat &flow2, cv::Mat &dist)
{
    assert(!flow1.empty() && flow1.type()==CV_32FC2 && flow2.type()==CV_32FC2 && flow1.size()==flow2.size());

    dist.create(flow1.size(), CV_32F);

    float avgDist = 0;
    float d;

    for (int y=0; y<flow1.rows; y++)
        for (int x=0; x<flow1.cols; x++)
        {
            cv::Point2f a = flow1.at<cv::Point2f>(y,x);
            cv::Point2f b = flow2.at<cv::Point2f>(y,x);
            d = (a-b).dot(a-b);
            avgDist += d;
            dist.at<float>(y,x) = d;
        }
    avgDist /= flow1.cols*flow1.rows;
    return avgDist;
}

int main(int argc, char ** argv)
{
    cv::Mat img0, img1, gtflow, flow, flowRefined;

    string dataDir = "./data/";
    string outputDir = "./";
    string outputPath = outputDir + "refinedflow_0001.flo";
    
    img0 = cv::imread(dataDir + "frame_0001.jpg", cv::IMREAD_COLOR);
    if (img0.data==nullptr)
    {
        cout<<"Failed to read first image"<<endl;
        return -1;
    }
    img0.convertTo(img0, CV_32F, 1.0/255.0);

    img1 = cv::imread(dataDir + "frame_0002.jpg", cv::IMREAD_COLOR);
    if (img1.data==nullptr)
    {
        cout<<"Failed to read second image"<<endl;
        return -1;
    }
    img1.convertTo(img1, CV_32F, 1.0/255.0);

    gtflow = cv::readOpticalFlow(dataDir + "gtflow_0001.flo");
    if (gtflow.data==nullptr)
    {
        cout<<"Failed to read ground truth flow"<<endl;
        return -1;
    }
        
    flow = cv::readOpticalFlow(dataDir + "dirtyflow_0001.flo");
    if (flow.data==nullptr)
    {
        cout<<"Failed to read input flow"<<endl;
        return -1;
    }
    
    DeepFlowCuda deepFlow;
    
    flowRefined = flow.clone();

    deepFlow.computeFlow(img0, img1, flowRefined);

    cv::Mat dist;
    cout<<"Distance = "<<L2Distance(flow, gtflow, dist)<<endl;
    cout<<"Distance refined = "<<L2Distance(flowRefined, gtflow, dist)<<endl;

    if (cv::writeOpticalFlow(outputPath, flowRefined)==false)
    {
        cout<<"Failed to write refined flow to"<<outputPath<<endl;
        return -1;
    }
    else
        cout<<"Refined flow written to "<<outputPath<<endl;

    return 0;
}
