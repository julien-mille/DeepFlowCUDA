This is a C++/OpenCV/CUDA port of the DeepFlow C library, associated to the following paper:
[P. Weinzaepfel, J. Revaud, Z. Harchaoui and C. Schmid. DeepFlow: Large displacement optical flow with deep matching. ICCV 2013]
See http://lear.inrialpes.fr/src/deepflow/

The library is provided with two images and the corresponding ground truth flow from the MPI Sintel dataset:
[D.J. Butler, J. Wulff, G.B. Stanley and M.J. Black. A naturalistic open source movie for optical flow evaluation. ECCV 2012]
See http://sintel.is.tue.mpg.de/

It should be used as a refinement step, as described for example in the FlowNet paper,
[A. Dosovitskiy, P. Fischer, E. Ilg, P. Häusser, C. Hazirbas, V. Golkov, P. van der Smagt, D. Cremers and T. Brox. FlowNet: Learning Optical Flow with Convolutional Networks. ICCV 2015]
Thus, the input is made up of two images and a coarse optical flow, typically output from a deep neural network. If no input flow is provided, the initial field is set to zero everywhere. As in the OpenCV modules variationalrefinement and cudaoptflow, the red-black successive overrelaxation method (SOR) is used to solve the linear systems. See the paper by Brox et al:
[T. Brox, A. Bruhn, N. Papenberg, J. Weickert. High Accuracy Optical Flow Estimation Based on a Theory for Warping. ECCVV 2004]

The DeepFlowCUDA class uses the cv::cudev::GpuMat_ template class, so you'll need OpenCV with additional opencv_contrib modules built (see https://github.com/opencv/opencv_contrib)

Requirements:
- OpenCV with opencv_contrib modules (need CUDA modules)
- CUDA build environment (nvcc should be in your PATH)

The Makefile assumes that CUDA is installed in /usr/local/cuda-9.2 and OpenCV in /usr/local/opencv-4.1.0
Just edit the paths in the Makefile and run
> make
> ./deepflow

Please report any bug to julien.mille@insa-cvl.fr
Thanks!

Copyright 2019 Julien Mille
