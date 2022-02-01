// This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// OpenCL includes
#include <CL/cl.h>

// Custom header file includes 
#include "../include/utils_yh.h"

int main() {
	// This code executes on the OpenCL host
	//				  N,C,H,W  K,P,Q, KH,KW, SH,SW, L,R,T,B
	Conv2dConfig c = { 1,3,1024,1024, 4,0,0, 2, 2,  1, 1,  0,0,0,0 };
	c.P = ((c.H + c.PT + c.PB - c.KH) / c.SH) + 1;
	c.Q = ((c.W + c.PL + c.PR - c.KW) / c.SW) + 1;
	printf(" input[%4d,%4d,%4d,%4d] kernel[%4d,%4d,%4d,%4d] output[%4d,%4d,%4d,%4d]\n\n", c.N, c.C, c.H, c.W, c.K, c.C, c.KH, c.KW, c.N, c.K, c.P, c.Q);

	// Host data
	float *data = NULL;		// Input matrix
	float *weight = NULL;	// Input matrix
	float *im2col = NULL;	// Input matrix
	float *output_m = NULL;	// Device Output matrix
	float *output_o = NULL;	// Device Output matrix
	float *output_h = NULL;	// Host Output matrix

	// Allocate space for input/output data
	data = (float*)malloc(sizeof(float)* c.N * c.C * c.H * c.W);		// input data [N,C,H,W]
	weight = (float*)malloc(sizeof(float) * c.K * c.C * c.KH * c.KW);	// weight [K,C,KH,KW]
	im2col = (float*)malloc(sizeof(float) * c.C * c.KH * c.KW * c.N * c.P * c.Q);	// im2col output 
	output_m = (float*)calloc(c.K * c.P * c.Q * c.N, sizeof(float)); // (�Ҵ�� ������ ���� 0 �ʱ�ȭ) 
	output_o = (float*)calloc(c.K * c.P * c.Q * c.N, sizeof(float)); // (�Ҵ�� ������ ���� 0 �ʱ�ȭ) 
	output_h = (float*)calloc(c.K * c.P * c.Q * c.N, sizeof(float)); // ���

	// input data �ʱ�ȭ
	//initDataStep(data, c.N * c.C * c.H * c.W);
	//initDataStep(weight, c.K * c.C * c.KH * c.KW);

	initDataRandom(data, c.N * c.C * c.H * c.W);
	initDataRandom(weight, c.K * c.C * c.KH * c.KW);

	//printf("Data(Input) \n");
	//printData(data, c.N, c.C, c.H, c.W, 1);			//�Է°� Ȯ��
	//printf("weight \n");
	//printData(weight, c.K, c.C, c.KH, c.KW, 1);		// ����ġ Ȯ��
	//printData(weight, 1, 1, c.K, c.C * c.KH * c.KW);// ����ġ Ȯ��

	convolution(output_h, data, weight, c.N, c.C, c.H, c.W, c.K, c.KH, c.KW, c.SH, c.SW);
	//printf("ouptut \n");
	//printData(output_h, c.N, c.K, c.P, c.Q, 1);	// ��� Ȯ��

	//================================================================================================================
	// �÷���, ����̽�, ���ؽ�Ʈ, Ŀ�ǵ� ť ���� �κ� (openCL �ڵ忡�� ���� �κ�)
	cl_int status;
	cl_platform_id *platforms = NULL;
	cl_uint numPlatforms = 0;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);							// ��� ������ �÷��� �� Ȯ��
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);	// �÷��� ������ ������ �� �ֵ��� ���� �Ҵ�
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);					// �÷��� ������ ������

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

	status = clGetDeviceIDs(platforms[platformNum_ - 1], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);	// ������ ����̽� ������ ������
	cl_context context = clCreateContext(NULL, 1, &devices[deviceNum_ - 1], NULL, NULL, &status);			// context ���� �� (���ϴ�)����̽��� ����
	cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[deviceNum_ - 1], CL_QUEUE_PROFILING_ENABLE, &status);			// ���ɾ� ť ���� �� (���ϴ�)����̽��� ����

	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================
	//-----------------------------------------------------
	// STEP 5: Create device buffers
	//----------------------------------------------------- 

	cl_mem buffer_I;		// Input data array on the device
	cl_mem buffer_W;		// Input weight array on the device
	cl_mem buffer_im2col;	// Output im2col array on the device
	cl_mem buffer_M;		// Output array on the device
	cl_mem buffer_O;		// Output array on the device

	buffer_I = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * c.N * c.C * c.H * c.W, NULL, &status);	// ����̽� ���� ��ü ����(�Է¿�)
	buffer_W = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * c.K * c.C * c.KH * c.KW, NULL, &status);	// ����̽� ���� ��ü ����(�Է¿�)
	buffer_im2col = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * c.C * c.KH * c.KW * c.N * c.P * c.Q, NULL, &status);	// ����̽� ���� ��ü ����(�Է¿�)
	buffer_M = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * c.K * c.P * c.Q * c.N, NULL, &status);	// ����̽� ���� ��ü ����(��¿�)
	buffer_O = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * c.K * c.P * c.Q * c.N, NULL, &status);	// ����̽� ���� ��ü ����(��¿�)

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 6: Write host data to device buffers
	//----------------------------------------------------- 

	status = clEnqueueWriteBuffer(cmdQueue, buffer_I, CL_FALSE, 0, sizeof(float) * c.N * c.C * c.H * c.W, data, 0, NULL, NULL); // host (A) -> device (bufferA)����
	status = clEnqueueWriteBuffer(cmdQueue, buffer_W, CL_FALSE, 0, sizeof(float) * c.K * c.C * c.KH* c.KW, weight, 0, NULL, NULL); // host (B) -> device (bufferB)����

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 7: Create and compile the program
	//----------------------------------------------------- 

	char* programSource = readSource("bicubic.cl"); // Ŀ�� �Լ� ���� �ε�
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);	// ���α׷� ����

	status = clBuildProgram(program, numDevices, &devices[deviceNum_ - 1], NULL, NULL, NULL);					// ����̽��� ���� ���α׷��� ����(������)

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 8: Create the kernel
	//----------------------------------------------------- 

	cl_kernel im2col_kernel = NULL;
	cl_kernel matMul_kernel = NULL;
	cl_kernel col2im_kernel = NULL;
	im2col_kernel = clCreateKernel(program, "im2col_kernel", &status); // Ŀ�� ���� (Ŀ�� �Լ� �̸��� ���ڷ� ����)
	matMul_kernel = clCreateKernel(program, "matMul_kernel", &status); // Ŀ�� ���� (Ŀ�� �Լ� �̸��� ���ڷ� ����)
	col2im_kernel = clCreateKernel(program, "col2im_kernel", &status); // Ŀ�� ���� (Ŀ�� �Լ� �̸��� ���ڷ� ����)

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 9: Set the kernel arguments
	//----------------------------------------------------- 
	status = clSetKernelArg(im2col_kernel, 0, sizeof(cl_mem), &buffer_im2col); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 1, sizeof(cl_mem), &buffer_I); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 2, sizeof(cl_int), &c.N); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 3, sizeof(cl_int), &c.K); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 4, sizeof(cl_int), &c.P); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 5, sizeof(cl_int), &c.Q); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 6, sizeof(cl_int), &c.C); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 7, sizeof(cl_int), &c.H); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 8, sizeof(cl_int), &c.W); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 9, sizeof(cl_int), &c.KH); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 10, sizeof(cl_int), &c.KW); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 11, sizeof(cl_int), &c.SH); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 12, sizeof(cl_int), &c.SW); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 13, sizeof(cl_int), &c.PL); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(im2col_kernel, 14, sizeof(cl_int), &c.PT); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)

	int k_v = c.C * c.KH* c.KW;
	int n_v = c.P * c.Q * c.N;
	status = clSetKernelArg(matMul_kernel, 0, sizeof(cl_mem), &buffer_W); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(matMul_kernel, 1, sizeof(cl_mem), &buffer_im2col); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(matMul_kernel, 2, sizeof(cl_mem), &buffer_M); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(matMul_kernel, 3, sizeof(cl_int), &c.K); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(matMul_kernel, 4, sizeof(cl_int), &n_v); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(matMul_kernel, 5, sizeof(cl_int), &k_v); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)

	status = clSetKernelArg(col2im_kernel, 0, sizeof(cl_mem), &buffer_O); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(col2im_kernel, 1, sizeof(cl_mem), &buffer_M); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(col2im_kernel, 2, sizeof(cl_mem), &c.N); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(col2im_kernel, 3, sizeof(cl_int), &c.K); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(col2im_kernel, 4, sizeof(cl_int), &c.P); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(col2im_kernel, 5, sizeof(cl_int), &c.Q); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 10: Configure the work-item structure
	//----------------------------------------------------- 

	size_t globalWorkSize[1]; // ������ ���� ��ũ �������� �ε��� ����(�۷ι� ��ũ ������) ����
	globalWorkSize[0] = c.C * c.KH * c.KW * c.N * c.P * c.Q;

	size_t globalWorkSize2[1]; // ������ ���� ��ũ �������� �ε��� ����(�۷ι� ��ũ ������) ����
	globalWorkSize2[0] = c.K * c.N * c.P * c.Q;

	size_t globalWorkSize3[1]; // ������ ���� ��ũ �������� �ε��� ����(�۷ι� ��ũ ������) ����
	globalWorkSize3[0] = c.K * c.N * c.P * c.Q;

	//-----------------------------------------------------
	// STEP 11: Enqueue the kernel for execution
	//----------------------------------------------------- 

	cl_event event;

	status = clEnqueueNDRangeKernel(cmdQueue, im2col_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event); // Ŀ�� ����
	status = clEnqueueNDRangeKernel(cmdQueue, matMul_kernel, 1, NULL, globalWorkSize2, NULL, 0, NULL, &event); // Ŀ�� ����
	status = clEnqueueNDRangeKernel(cmdQueue, col2im_kernel, 1, NULL, globalWorkSize3, NULL, 0, NULL, &event); // Ŀ�� ����
	clWaitForEvents(3, &event);
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

	status = clEnqueueReadBuffer(cmdQueue, buffer_im2col, CL_TRUE, 0, sizeof(float) * c.C * c.KH * c.KW * c.N * c.P * c.Q, im2col, 0, NULL, NULL); // device (bufferC) -> host (C) ����
	status = clEnqueueReadBuffer(cmdQueue, buffer_M, CL_TRUE, 0, sizeof(float) * c.K * c.P * c.Q * c.N, output_m, 0, NULL, NULL); // device (bufferC) -> host (C) ����
	status = clEnqueueReadBuffer(cmdQueue, buffer_O, CL_TRUE, 0, sizeof(float) * c.K * c.P * c.Q * c.N, output_o, 0, NULL, NULL); // device (bufferC) -> host (C) ����

	//printData(im2col, 1, 1, c.C * c.KH * c.KW, c.N * c.P * c.Q);
	//printData(output_m, 1, 1, c.K, c.N * c.P * c.Q);
	//printData(output_o, c.N, c.K, c.P, c.Q);

	compareResults(output_h, output_o, c.K * c.P * c.Q * c.N); // Verify the output

	if (status != 0) checkError(status, __LINE__);
	//-----------------------------------------------------
	// STEP 13: Release OpenCL resources
	//----------------------------------------------------- 

	// Free OpenCL resources
	clReleaseKernel(im2col_kernel);
	clReleaseKernel(matMul_kernel);
	clReleaseKernel(col2im_kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(buffer_I);
	clReleaseMemObject(buffer_W);
	clReleaseMemObject(buffer_im2col);
	clReleaseMemObject(buffer_M);
	clReleaseMemObject(buffer_O);
	clReleaseContext(context);

	// Free host resources
	free(platforms);
	free(devices);
	free(data);
	free(weight);
	free(output_m);
	free(output_o);
	free(output_h);

	return 0;
}