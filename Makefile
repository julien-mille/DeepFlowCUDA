NVCC = nvcc
NVCCFLAGS = -c -g -O3
LD = g++
LDFLAGS = -O3
FINAL_TARGET = deepflow
CUDA_DIR = /usr/local/cuda-9.2
OPENCV_DIR = /usr/local/opencv-4.1.0-build
INCLUDE_DIR = -I$(CUDA_DIR)/include -I$(OPENCV_DIR)/include/opencv4
LIB_DIR = -L$(CUDA_DIR)/lib64 -L$(OPENCV_DIR)/lib
LIBS =  -lopencv_cudawarping -lopencv_cudafilters -lopencv_cudaimgproc -lopencv_cudaarithm -lopencv_cudalegacy -lopencv_video -lopencv_imgproc -lopencv_imgcodecs -lopencv_core -lcudart

default: $(FINAL_TARGET)

$(FINAL_TARGET): main.o deepflowcuda.o
	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_DIR) $< -o $@

clean:
	rm -f *.o $(FINAL_TARGET)

