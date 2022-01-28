// Matrix multiplication (Tiling in the local memory)

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// OpenCL includes
#include <CL/cl.h>

// Custom header file includes
#include "../include/utils_yh.h"

#define TS 16			// The square-root of the 2D tile-size (== work-group dims)
#define WPT 8			// The amount of work-per-thread, i.e. the thread-coarsening factor
#define RTS (TS/WPT)	// The reduced tile-size in one dimension

// platform 1. Device 1: Intel(R) UHD Graphics 750
// Program file is : MatMul2.cl
// dur_time(openCL) = 27.68108[msec]
// dur_time(cpu) = 1102.48499[msec]

int main() {
	// This code executes on the OpenCL host
	// A[M, K] * B[K, N] = C[M, N]
	const int M = 1024;
	const int K = 1024;
	const int N = 1024;

	// Host data
	float *A = NULL;  // Input matrix
	float *B = NULL;  // Input matrix
	float *C = NULL;  // Device Output matrix
	float *H = NULL;  // Host Output matrix

	// Allocate space for input/output data
	A = (float*)malloc(sizeof(float) * M * K);
	B = (float*)malloc(sizeof(float) * K * N);
	C = (float*)malloc(sizeof(float) * M * N);
	H = (float*)malloc(sizeof(float) * M * N);

	// input data 초기화
	initDataScalar(A, M * K);
	initDataScalar(B, K * N);

	//printData(A, M, K);
	//printData(B, K, N);

	//================================================================================================================
	// 플랫폼, 디바이스, 컨텍스트, 커맨드 큐 설정 부분 (openCL 코드에서 공통 부분)
	cl_int status;
	cl_platform_id *platforms = NULL;
	cl_uint numPlatforms = 0;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);							// 사용 가능한 플랫폼 수 확인
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);	// 플랫폼 정보를 가져올 수 있도록 공간 할당
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);					// 플랫폼 정보를 가져옴

	cl_uint numDevices = 0;
	cl_device_id * devices = NULL;
	char* value;
	size_t valueSize;
	cl_uint maxComputeUnits;

	for (int i = 0; i < numPlatforms; i++) {
		// get all devices
		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
		// for each device print critical attributes
		for (int j = 0; j < numDevices; j++) {
			// print device name
			status = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			status = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
			printf("platform %d. Device %d: %s\n", i + 1, j + 1, value);
			free(value);
			// print hardware device version
			status = clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			status = clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
			printf(" %d.%d Hardware version: %s\n", i + 1, 1, value);
			free(value);
			// print software driver version
			status = clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			status = clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
			printf(" %d.%d Software version: %s\n", i + 1, 2, value);
			free(value);
			// print c version supported by compiler for device
			status = clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			status = clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
			printf(" %d.%d OpenCL C version: %s\n", i + 1, 3, value);
			free(value);
			// print parallel compute units
			status = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
			printf(" %d.%d Parallel compute units: %d\n", i + 1, 4, maxComputeUnits);
		}
	}

	int platformNum_;
	int deviceNum_;
	bool selection = false;

	if (selection) {
		printf("\n\nSELECT PLATFORM('1' ~ '%d') : ", numPlatforms);
		scanf("%d", &platformNum_);
		printf("\n");
		printf("SELECT DEVICE('1' ~ '%d') : ", numDevices);
		scanf("%d", &deviceNum_);
		printf("\n");
	}
	else {
		platformNum_ = 1;
		deviceNum_ = 1;
	}

	status = clGetDeviceIDs(platforms[platformNum_ - 1], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);	// 선택한 디바이스 정보를 가져옴
	cl_context context = clCreateContext(NULL, 1, &devices[deviceNum_ - 1], NULL, NULL, &status);			// context 생성 및 (원하는)디바이스와 연결
	cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[deviceNum_ - 1], CL_QUEUE_PROFILING_ENABLE, &status);			// 명령어 큐 생성 및 (원하는)디바이스와 연결

	//================================================================================================================

	//-----------------------------------------------------
	// STEP 5: Create device buffers
	//----------------------------------------------------- 

	cl_mem bufferA;  // Input array on the device
	cl_mem bufferB;  // Input array on the device
	cl_mem bufferC;  // Output array on the device
	bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * M * K, NULL, &status);	// 디바이스 버퍼 객체 생성(입력용)
	bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * K * N, NULL, &status);	// 디바이스 버퍼 객체 생성(입력용)
	bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, NULL, &status);	// 디바이스 버퍼 객체 생성(출력용)

	//-----------------------------------------------------
	// STEP 6: Write host data to device buffers
	//----------------------------------------------------- 

	status = clEnqueueWriteBuffer(cmdQueue, bufferA, CL_FALSE, 0, sizeof(float) * M * K, A, 0, NULL, NULL); // host (A) -> device (bufferA)전달
	status = clEnqueueWriteBuffer(cmdQueue, bufferB, CL_FALSE, 0, sizeof(float) * K * N, B, 0, NULL, NULL); // host (B) -> device (bufferB)전달

	//-----------------------------------------------------
	// STEP 7: Create and compile the program
	//----------------------------------------------------- 

	char* programSource = readSource("MatMul3.cl"); // 커널 함수 파일 로드
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);	// 프로그램 생성
	status = clBuildProgram(program, numDevices, &devices[deviceNum_ - 1], NULL, NULL, NULL);					// 디바이스를 위한 프로그램을 빌드(컴파일)

	//-----------------------------------------------------
	// STEP 8: Create the kernel
	//----------------------------------------------------- 

	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "wpt_matMul_kernel", &status); // 커널 생성 (커널 함수 이름을 인자로 전달)

	//-----------------------------------------------------
	// STEP 9: Set the kernel arguments
	//----------------------------------------------------- 

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(kernel, 3, sizeof(cl_int), &M); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(kernel, 4, sizeof(cl_int), &N); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(kernel, 5, sizeof(cl_int), &K); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)

	//-----------------------------------------------------
	// STEP 10: Configure the work-item structure
	//----------------------------------------------------- 

	const size_t local[2] = { TS, TS / WPT };
	const size_t global[2] = { (size_t)M, (size_t)(N / WPT) };

	//-----------------------------------------------------
	// STEP 11: Enqueue the kernel for execution
	//----------------------------------------------------- 

	cl_event event;
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, global, local, 0, NULL, &event); // 커널 실행
	checkError(status, __LINE__);

	clWaitForEvents(1, &event);
	clFinish(cmdQueue);

	cl_ulong time_start;
	cl_ulong time_end;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	printf("dur_time(openCL)           = %6.5f [msec] \n", (time_end - time_start) / 1000000.0);

	//-----------------------------------------------------
	// STEP 12: Read the output buffer back to the host
	//----------------------------------------------------- 

	status = clEnqueueReadBuffer(cmdQueue, bufferC, CL_TRUE, 0, sizeof(float) * M * N, C, 0, NULL, NULL); // device (bufferC) -> host (C) 전달

	matMulCpu(A, B, H, M, N, K);
	//printData(H, M, N);

	compareResults(H, C, M * N); // Verify the output
	//printData(C, M, N);

	//-----------------------------------------------------
	// STEP 13: Release OpenCL resources
	//----------------------------------------------------- 

	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseContext(context);

	// Free host resources
	free(A);
	free(B);
	free(C);
	free(H);
	free(platforms);
	free(devices);
}