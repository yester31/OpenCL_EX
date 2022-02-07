// This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
#include <CL/cl.h>

// Custom header file includes 
#include "../include/utils_yh.h"

int main() {
	// Bicubic Interpolation
	// Input [N, C, H, W] -> Ouput [N, C, P, Q]
	float rescale_factor = 2.f;
	int N = 1;
	int C = 3;
	int H = 1080;
	int W = 1920;
	int P = H * rescale_factor;
	int Q = W * rescale_factor;
	float h_scale = float(H - 1) / (P - 1);
	float w_scale = float(W - 1) / (Q - 1);
	int num_elements = N * C * P * Q;
	printf("input[%4d,%4d,%4d,%4d] -> output[%4d,%4d,%4d,%4d]\n\n", N, C, H, W, N, C, P, Q);

	// Host data
	float *input = NULL;		// Input matrix
	float *output = NULL;	// Host Output matrix

	// Allocate space for input/output data
	input = (float*)malloc(sizeof(float)* N * C * H * W);		// input [N, C, H, W]
	output = (float*)malloc(sizeof(float)* N * C * P * Q);		// Ouput [N, C, P, Q]

	// input data 초기화
	initDataRandom255(input, N * C * H * W);

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

	cl_mem buffer_I;		// Input data array on the device
	cl_mem buffer_O;		// Output array on the device
	buffer_I = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * N * C * H * W, NULL, &status);		// 디바이스 버퍼 객체 생성(입력용)
	buffer_O = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * C * P * Q, NULL, &status);	// 디바이스 버퍼 객체 생성(출력용)

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 6: Write host data to device buffers
	//----------------------------------------------------- 

	status = clEnqueueWriteBuffer(cmdQueue, buffer_I, CL_FALSE, 0, sizeof(float) * N * C * H * W, input, 0, NULL, NULL); // host (input) -> device (buffer_I)전달

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 7: Create and compile the program
	//----------------------------------------------------- 

	char* programSource = readSource("bicubic.cl"); // 커널 함수 파일 로드
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);	// 프로그램 생성
	status = clBuildProgram(program, numDevices, &devices[deviceNum_ - 1], NULL, NULL, NULL);					// 디바이스를 위한 프로그램을 빌드(컴파일)

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 8: Create the kernel
	//----------------------------------------------------- 

	cl_kernel bicubic_kernel = NULL;
	bicubic_kernel = clCreateKernel(program, "bicubic2d_kernel", &status); // 커널 생성 (커널 함수 이름을 인자로 전달)

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 9: Set the kernel arguments
	//----------------------------------------------------- 

	// 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(bicubic_kernel, 0, sizeof(cl_mem), &buffer_O); // output
	status = clSetKernelArg(bicubic_kernel, 1, sizeof(cl_mem), &buffer_I); // input
	status = clSetKernelArg(bicubic_kernel, 2, sizeof(cl_float), &h_scale); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(bicubic_kernel, 3, sizeof(cl_float), &w_scale); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(bicubic_kernel, 4, sizeof(cl_int), &N); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(bicubic_kernel, 5, sizeof(cl_int), &C); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(bicubic_kernel, 6, sizeof(cl_int), &H); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(bicubic_kernel, 7, sizeof(cl_int), &W); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(bicubic_kernel, 8, sizeof(cl_int), &P); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(bicubic_kernel, 9, sizeof(cl_int), &Q); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(bicubic_kernel, 10, sizeof(cl_int), &num_elements); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 10: Configure the work-item structure
	//----------------------------------------------------- 

	size_t globalWorkSize[1]; // 실행을 위한 워크 아이템의 인덱스 공간(글로벌 워크 사이즈) 정의
	globalWorkSize[0] = num_elements;

	//-----------------------------------------------------
	// STEP 11: Enqueue the kernel for execution
	//----------------------------------------------------- 

	cl_event event;

	status = clEnqueueNDRangeKernel(cmdQueue, bicubic_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event); // 커널 실행
	clWaitForEvents(1, &event);
	clFinish(cmdQueue);

	cl_ulong time_start;
	cl_ulong time_end;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	printf("dur_time(openCL)           = %6.5f [msec] \n", (time_end - time_start) / 1000000.0);

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 12: Read the output buffer back to the host
	//----------------------------------------------------- 

	status = clEnqueueReadBuffer(cmdQueue, buffer_O, CL_TRUE, 0, sizeof(float) * N * C * P * Q, output, 0, NULL, NULL); // device (bufferC) -> host (C) 전달

	//printData(output, N, C, P, Q);
	tofile(input, N * C * H * W, "../Validation_py/Input_C");
	tofile(output, N * C * P * Q, "../Validation_py/Output_C");

	// python 검증 스크립트 수행
	printf("\n *Validation with python \n");
	std::string command = "python ../Validation_py/bicubic.py --N=" + std::to_string(N) + " --C=" + std::to_string(C) + " --H=" + std::to_string(H) + " --W=" + std::to_string(W);
	const char *cmd = command.c_str();
	system(cmd); //터미널에 명령어 전달 


	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 13: Release OpenCL resources
	//----------------------------------------------------- 

	// Free OpenCL resources
	clReleaseKernel(bicubic_kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(buffer_I);
	clReleaseMemObject(buffer_O);
	clReleaseContext(context);

	// Free host resources
	free(platforms);
	free(devices);
	free(input);
	free(output);

	return 0;
}