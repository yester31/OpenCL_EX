# OpenCL_EX

## Enviroments
* Windows 10 laptop
* CPU 11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz (cpu)
* Intel(R) Iris(R) Xe Graphics (iGPU)
* NVIDIA GeForce RTX 3060 Laptop GPU (gpu)

## OpenCL Vector Add(Completed)
- vecAdd

##  OpenCL Matrix Multiplication(in progress)
- MatrixMultiplication project.
- Time check (wo : without data transfer time for Device)
     - A[1024, 1024] * B[1024, 1024] = C[1024, 1024]
  
1. Naive Matrix Multiplication
   - main.cpp with MatMul.cl
   - CUDA Kernel   = 4.982 [msec]
   - OpenCL Kernel = 5.587 [msec]
     
2. Matrix Multiplication Tiling in the local memory
   - main2.cpp with MatMul2.cl
3. Matrix Multiplication Tiling in the local memory with register level
   - main3.cpp with MatMul3.cl
4. Register Blocking Matrix Multiplication
   - main4.cpp with MatMul4.cl


##  OpenCL Convolution(in progress)
1. GEMM Convolution
   - process : im2col -> Matrix Multiplication -> col2im
2. Conventional Convolution
   - a



##  OpenCL Bicubic Interpolation(Preparing)
*





##  OpenCL Sparse Matrix Multiplication(Preparing)
*




## Reference
