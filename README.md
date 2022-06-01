# OpenCL_EX

## Enviroments
* Windows 10 laptop
* CPU 11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz (cpu)
* Intel(R) Iris(R) Xe Graphics (iGPU)
* NVIDIA GeForce RTX 3060 Laptop GPU (gpu)

## OpenCL Vector Add
- vector sum


##  OpenCL Matrix Multiplication(in progress)
- MatrixMultiplication project
  1. Naive Matrix Multiplication (Completed)
     - main.cpp with MatMul.cl
     - No limit on input size
     
  2. Matrix Multiplication Tiling in the local memoryv (Completed)
     - main2.cpp with MatMul2.cl
     - No limit on input size
   
  3. Matrix Multiplication Tiling in the local memory with register level(Verification Required)
     - main3.cpp with MatMul3.cl

  4. Register Blocking Matrix Multiplication(Verification Required)
     - main4.cpp with MatMul4.cl
  
  - Time check for A[1024, 1024] * B[1024, 1024] = C[1024, 1024] (wo : without data transfer time for Device)


##  OpenCL Convolution(in progress)
1. GEMM Convolution(Completed)
   - process : im2col -> Matrix Multiplication -> col2im
   - Matrix Multiplication can be changed to better logic
2. Naive Convolution(Completed)
3. FFT Convolution(Plan)
4. Winograd Convolution(Plan)

	- Time Check (input[1, 3, 512, 512] kernel[3, 3, 3, 3] output[ 1, 3, 510, 510])
		- GEMM Conv2d  (gpu) : 1.76237 [msec]
		- Naive Conv2d (gpu) : 0.22528 [msec]
		- Naive Conv2d (cpu) : 22.32400 [msec]

##  OpenCL Bicubic Interpolation
- Bicubic Interpolation
- Add test code(validation results with pytorch Bicubic Interpolation) 


##  OpenCL Nearnest Neighbor Interpolation
- Nearnest Neighbor Interpolation
- Add test code(validation results with pytorch Nearnest Neighbor Interpolation) 


##  OpenCL BGR2YCbCr
- transformation image data format BGR to YCbC, NHWC->NCHW
- Add test code(validation results with python BGR2YCbC) 


##  OpenCL YCbCr2BGR
- transformation image data format YCbC to BGR, NCHW->NHWC
- Add test code(validation results with python YCbC2BGR)


##  OpenCL Concat
- Concatenate two tensor for channal side
- Add test code(validation results with python concat)



## Reference
