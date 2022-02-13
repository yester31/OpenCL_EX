// This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
#include <CL/cl.h>

// Custom header file includes 
#include "utils_cl.h"
#include "layers.h"

int main() {
	// Bicubic Interpolation
	// Input [N, C, H, W] -> Ouput [N, C, P, Q]
	//Configs ccfg_b(1, 3, 1080, 1920, 2); // bicubic
	Configs ccfg_b(1, 3, 256, 256, 2); // bicubic

	// host�������� ����� ������ ���� �Ҵ�
	uint8_t *input = (uint8_t*)malloc(sizeof(uint8_t)* ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W);// input data [N,C,H,W]
	float *output = (float*)malloc(sizeof(float)* ccfg_b.N * ccfg_b.K * ccfg_b.P * ccfg_b.Q);		// Ouput [N, C, P, Q]
	//================================================================================================================

	// input data �ʱ�ȭ
	initDataRandom255(input, ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W);
	//================================================================================================================

	// �÷���, ����̽�, ���ؽ�Ʈ, Ŀ�ǵ� ť ���� �κ� (openCL �ڵ忡�� ���� �κ�)
	ocl_args_d_t ocl;
	cl_int status;
	size_t source_size = 0;
	char* clKernelSource = readSource("Template.cl", source_size); // OpenCL Ŀ�� �ҽ� ���� �ε�
	cl_setting(ocl, status, clKernelSource, source_size, true, true);
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// device���� ����� �޸� ���� �Ҵ�
	cl_mem inputBuffer_b = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(uint8_t) * ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W, NULL, &status);	// Input data array on the device
	cl_mem outputBuffer_b = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float) * ccfg_b.N * ccfg_b.K * ccfg_b.P * ccfg_b.Q, NULL, &status);	// Output array on the device
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// host�����͸� device�� ���� (host -> device)
	status = clEnqueueWriteBuffer(ocl.commandQueue, inputBuffer_b, CL_FALSE, 0, sizeof(uint8_t) * ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W, input, 0, NULL, NULL); 
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	bool time = false;
	if (1) {
		//for (int i = 0; i < 10000; i++) {
		bicubic_opencl(outputBuffer_b, inputBuffer_b, ccfg_b, ocl, status, time);
		//}
	}
	else {
		// Ŀ�� ���� (Ŀ�� �Լ� �̸��� ���ڷ� ����)
		cl_kernel bicubic_kernel = clCreateKernel(ocl.program, "bicubic_kernel", &status);
		if (status != 0) checkError(status, __LINE__);
		//================================================================================================================

		// Ŀ�ο��� ����� ���� �� ��� �۾�
		float h_scale = float(ccfg_b.H - 1) / (ccfg_b.P - 1);
		float w_scale = float(ccfg_b.W - 1) / (ccfg_b.Q - 1);
		int num_elements = ccfg_b.N * ccfg_b.C * ccfg_b.P * ccfg_b.Q;
		//================================================================================================================

		// Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
		status = clSetKernelArg(bicubic_kernel, 0, sizeof(cl_mem), &outputBuffer_b); // output
		status = clSetKernelArg(bicubic_kernel, 1, sizeof(cl_mem), &inputBuffer_b); // input
		status = clSetKernelArg(bicubic_kernel, 2, sizeof(cl_float), &h_scale);
		status = clSetKernelArg(bicubic_kernel, 3, sizeof(cl_float), &w_scale);
		status = clSetKernelArg(bicubic_kernel, 4, sizeof(cl_int), &ccfg_b.N);
		status = clSetKernelArg(bicubic_kernel, 5, sizeof(cl_int), &ccfg_b.C);
		status = clSetKernelArg(bicubic_kernel, 6, sizeof(cl_int), &ccfg_b.H);
		status = clSetKernelArg(bicubic_kernel, 7, sizeof(cl_int), &ccfg_b.W);
		status = clSetKernelArg(bicubic_kernel, 8, sizeof(cl_int), &ccfg_b.P);
		status = clSetKernelArg(bicubic_kernel, 9, sizeof(cl_int), &ccfg_b.Q);
		status = clSetKernelArg(bicubic_kernel, 10, sizeof(cl_int), &num_elements);
		if (status != 0) checkError(status, __LINE__);
		//================================================================================================================

		// ������ ���� ��ũ �������� �ε��� ����(�۷ι� ��ũ ������) ����
		size_t globalWorkSize[1]; 
		globalWorkSize[0] = num_elements;
		//================================================================================================================

		if (time) {
			// Ŀ�� ����, event�� �̿��� Ŀ�� ���� �ð� ����
			cl_event event;
			status = clEnqueueNDRangeKernel(ocl.commandQueue, bicubic_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event);
			if (status != 0) checkError(status, __LINE__);
			status = clWaitForEvents(1, &event);
			if (status != 0) checkError(status, __LINE__);
			status = clFinish(ocl.commandQueue);
			if (status != 0) checkError(status, __LINE__);
			cl_ulong time_start;
			cl_ulong time_end;
			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
			printf("dur_time(openCL)           = %6.5f [msec] \n", (time_end - time_start) / 1000000.0);
			//================================================================================================================
		}
		else {
			// Ŀ�� ����
			status = clEnqueueNDRangeKernel(ocl.commandQueue, bicubic_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
			if (status != 0) checkError(status, __LINE__);
			status = clFinish(ocl.commandQueue);
			if (status != 0) checkError(status, __LINE__);
			//================================================================================================================
		}
	}

	// device�� ��� ����� host�� �ҷ����� (device -> host)
	status = clEnqueueReadBuffer(ocl.commandQueue, outputBuffer_b, CL_TRUE, 0, sizeof(float) * ccfg_b.N * ccfg_b.C * ccfg_b.P * ccfg_b.Q, output, 0, NULL, NULL); // device (bufferC) -> host (C) ����
	if (status != 0) checkError(status, __LINE__);

	// valication with python results
	if (1) {
		//printData(output, N, C, P, Q);
		tofile(input, ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W, "../Validation_py/Input_C");
		tofile(output, ccfg_b.N * ccfg_b.C * ccfg_b.P * ccfg_b.Q, "../Validation_py/Output_C");

		// python ���� ��ũ��Ʈ ���� ��ɾ� ����
		printf("\n *Validation with python \n");
		std::string command = "python ../Validation_py/bicubic.py --N=" + std::to_string(ccfg_b.N) + " --C=" + std::to_string(ccfg_b.C) + " --H=" + std::to_string(ccfg_b.H) + " --W=" + std::to_string(ccfg_b.W);
		const char *cmd = command.c_str();
		// �͹̳ο� ��ɾ� ���� 
		system(cmd); 
	}

	// �ڿ� ����
	// Free OpenCL resources
	//clReleaseKernel(bicubic_kernel);
	clReleaseMemObject(inputBuffer_b);
	clReleaseMemObject(outputBuffer_b);

	// Free host resources
	free(input);
	free(output);

	return 0;
}