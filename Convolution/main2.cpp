// This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <assert.h>
// OpenCL includes
#include <CL/cl.h>

// Custom header file includes 
#include "../include/utils_yh.h"

#define TS 32			// The square-root of the 2D tile-size (== work-group dims)

int main() {
	// This code executes on the OpenCL host
	//				  N,C,H,W        K,P,Q, KH,KW, SH,SW, L,R,T,B
	//Conv2dConfig c = {1,3,1024,1024, 4,0,0, 2, 2,  1, 1,  0,0,0,0};
	//Conv2dConfig c = {1,1,2160,3840, 64,0,0, 9, 9,  1, 1,  4,4,4,4};
	//Conv2dConfig c = {1,1,2160,3840, 32,0,0, 3, 3,  1, 1,  2,2,2,2};
	//Conv2dConfig c = {1,1,2160,3840, 1,0,0, 5, 5,  1, 1,  2,2,2,2};
	Conv2dConfig c = { 1,3,512,512, 3,0,0, 3, 3,  1, 1,  0,0,0,0 };
	//Conv2dConfig c = {1,1,10,10, 64,0,0, 9, 9,  1, 1,  4,4,4,4};
	//Conv2dConfig c = {1,3,3,3, 2,0,0, 2, 2,  1, 1,  1,1,1,1 };
	c.P = ((c.H + c.PT + c.PB - c.KH) / c.SH) + 1;
	c.Q = ((c.W + c.PL + c.PR - c.KW) / c.SW) + 1;
	printf(" input[%4d,%4d,%4d,%4d] kernel[%4d,%4d,%4d,%4d] output[%4d,%4d,%4d,%4d]\n\n", c.N, c.C, c.H, c.W, c.K, c.C, c.KH, c.KW, c.N, c.K, c.P, c.Q);

	// Host data
	float *data = NULL;			// Input matrix
	float *data_pad = NULL;		// Input matrix (WITH PAD)
	float *weight = NULL;		// weight matrix
	float *bias = NULL;		// weight matrix
	float *output_o = NULL;	// Device Output matrix
	float *output_h = NULL;	// Host Output matrix

	// Allocate space for input/output data
	data = (float*)malloc(sizeof(float)* c.N * c.C * c.H * c.W);		// input data [N,C,H,W]
	weight = (float*)malloc(sizeof(float) * c.K * c.C * c.KH * c.KW);	// weight [K,C,KH,KW]
	bias = (float*)malloc(sizeof(float) * c.K);	// weight [K,C,KH,KW]
	data_pad = (float*)calloc(c.N * c.C * (c.H + c.PT + c.PB) * (c.W + c.PL + c.PR), sizeof(float)); // input data [N,C,H+PT+PB,W+PL+PR]
	output_o = (float*)calloc(c.K * c.P * c.Q * c.N, sizeof(float)); // (할당된 공간의 값을 0 초기화) 
	output_h = (float*)calloc(c.K * c.P * c.Q * c.N, sizeof(float)); // 결과

	// input data 초기화
	//initDataStep(data, c.N * c.C * c.H * c.W);
	//initDataStep(weight, c.K * c.C * c.KH * c.KW);

	initDataRandomZP1(data, c.N * c.C * c.H * c.W);
	initDataRandomZP1(weight, c.K * c.C * c.KH * c.KW);
	initDataRandomZP1(bias, c.K);
	printf("Initialize data completed\n");

	//printf("Data(Input) \n");
	//printData(data, c.N, c.C, c.H, c.W);			//입력값 확인
	//printf("weight \n");
	//printData(weight, c.K, c.C, c.KH, c.KW);		// 가중치 확인
	//printData(bias, 1, 1, 1, c.K);	// 가중치 확인

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
	int platformNum_;
	int deviceNum_;

	bool selection = false;
	if (selection) {
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
		printf("\n\nSELECT PLATFORM('1' ~ '%d') : ", numPlatforms);
		scanf("%d", &platformNum_);
		printf("\n");
		printf("SELECT DEVICE('1' ~ '%d') : ", numDevices);
		scanf("%d", &deviceNum_);
		printf("\n");
	}
	else {
		platformNum_ = 2; // 1 or 2 or 3
		deviceNum_ = 1;

		status = clGetDeviceIDs(platforms[platformNum_ - 1], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
		status = clGetDeviceIDs(platforms[platformNum_ - 1], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

		status = clGetDeviceInfo(devices[deviceNum_ - 1], CL_DEVICE_NAME, 0, NULL, &valueSize);
		value = (char*)malloc(valueSize);
		status = clGetDeviceInfo(devices[deviceNum_ - 1], CL_DEVICE_NAME, valueSize, value, NULL);
		printf("platform %d. Device %d: %s\n", platformNum_, deviceNum_, value);
		free(value);
	}

	status = clGetDeviceIDs(platforms[platformNum_ - 1], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);	// 선택한 디바이스 정보를 가져옴
	cl_context context = clCreateContext(NULL, 1, &devices[deviceNum_ - 1], NULL, NULL, &status);			// context 생성 및 (원하는)디바이스와 연결
	cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[deviceNum_ - 1], CL_QUEUE_PROFILING_ENABLE, &status);			// 명령어 큐 생성 및 (원하는)디바이스와 연결

	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================
	//-----------------------------------------------------
	// STEP 5: Create device buffers
	//----------------------------------------------------- 

	cl_mem 	buffer_I = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * c.N * c.C * c.H * c.W, NULL, &status);	// 디바이스 버퍼 객체 생성(입력용)
	cl_mem 	buffer_W = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * c.K * c.C * c.KH * c.KW, NULL, &status);	// 디바이스 버퍼 객체 생성(입력용)
	cl_mem 	buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * c.K, NULL, &status);	// 디바이스 버퍼 객체 생성(입력용)
	cl_mem 	buffer_O = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * c.K * c.P * c.Q * c.N, NULL, &status);	// 디바이스 버퍼 객체 생성(출력용)

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 6: Write host data to device buffers
	//----------------------------------------------------- 

	status = clEnqueueWriteBuffer(cmdQueue, buffer_I, CL_FALSE, 0, sizeof(float) * c.N * c.C * c.H * c.W, data, 0, NULL, NULL); // host (A) -> device (bufferA)전달
	status = clEnqueueWriteBuffer(cmdQueue, buffer_W, CL_FALSE, 0, sizeof(float) * c.K * c.C * c.KH* c.KW, weight, 0, NULL, NULL); // host (B) -> device (bufferB)전달
	status = clEnqueueWriteBuffer(cmdQueue, buffer_B, CL_FALSE, 0, sizeof(float) * c.K , bias, 0, NULL, NULL); // host (B) -> device (bufferB)전달

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 7: Create and compile the program
	//----------------------------------------------------- 

	char* programSource = readSource("Convolution.cl"); // 커널 함수 파일 로드
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);	// 프로그램 생성

	status = clBuildProgram(program, numDevices, &devices[deviceNum_ - 1], NULL, NULL, NULL);					// 디바이스를 위한 프로그램을 빌드(컴파일)

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 8: Create the kernel
	//----------------------------------------------------- 
	cl_kernel conv2d_kernel = clCreateKernel(program, "conv2d_kernel", &status); // 커널 생성 (커널 함수 이름을 인자로 전달)

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 9: Set the kernel arguments
	status = clSetKernelArg(conv2d_kernel, 0, sizeof(cl_mem), &buffer_O); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 1, sizeof(cl_mem), &buffer_I); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 2, sizeof(cl_mem), &buffer_W); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 3, sizeof(cl_mem), &buffer_B); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 4, sizeof(cl_int), &c.N); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 5, sizeof(cl_int), &c.C); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 6, sizeof(cl_int), &c.H); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 7, sizeof(cl_int), &c.W); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 8, sizeof(cl_int), &c.K); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 9, sizeof(cl_int), &c.P); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 10, sizeof(cl_int), &c.Q); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 11, sizeof(cl_int), &c.KH); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 12, sizeof(cl_int), &c.KW); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 13, sizeof(cl_int), &c.SH); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 14, sizeof(cl_int), &c.SW); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 15, sizeof(cl_int), &c.PL); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(conv2d_kernel, 16, sizeof(cl_int), &c.PT); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 10: Configure the work-item structure
	//----------------------------------------------------- 

	size_t globalWorkSize[1]; // 실행을 위한 워크 아이템의 인덱스 공간(글로벌 워크 사이즈) 정의
	globalWorkSize[0] = c.K * c.N * c.P * c.Q;

	//-----------------------------------------------------
	// STEP 11: Enqueue the kernel for execution
	//----------------------------------------------------- 
	cl_event event;
	status = clEnqueueNDRangeKernel(cmdQueue, conv2d_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event); // 커널 실행
	if (status != 0) checkError(status, __LINE__);
	clWaitForEvents(1, &event);
	clFinish(cmdQueue); //clFlush(cmdQueue);

	cl_ulong time_start;
	cl_ulong time_end;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	printf("dur_time(openCL)           = %6.5f [msec] \n", (time_end - time_start) / 1000000.0);

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 12: Read the output buffer back to the host
	//----------------------------------------------------- 

	status = clEnqueueReadBuffer(cmdQueue, buffer_O, CL_TRUE, 0, sizeof(float) * c.K * c.P * c.Q * c.N, output_o, 0, NULL, NULL); // device (bufferC) -> host (C) 전달
	clReleaseMemObject(buffer_O);

	//printData(output_o, c.N, c.K, c.P, c.Q);

	zeroPadding(data_pad, data, c.N, c.C, c.H, c.W, c.PL, c.PR, c.PT, c.PB);
	//printData(data_pad, c.N, c.C, (c.H + c.PT + c.PB), (c.W + c.PL + c.PR), 1);	// 결과 확인
	convolution(output_h, data_pad, weight, bias, c.N, c.C, (c.H + c.PT + c.PB), (c.W + c.PL + c.PR), c.K, c.KH, c.KW, c.SH, c.SW);
	//printf("cpu ouptut \n");
	//printData(output_h, c.N, c.K, c.P, c.Q);	// 결과 확인
	compareResults(output_h, output_o, c.K * c.P * c.Q * c.N); // Verify the output

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 13: Release OpenCL resources
	//----------------------------------------------------- 

	// Free OpenCL resources
	clReleaseKernel(conv2d_kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseContext(context);

	// Free host resources
	free(platforms);
	free(devices);
	free(data);
	free(weight);
	free(output_o);
	free(output_h);

	return 0;
}