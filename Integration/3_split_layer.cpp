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
	Configs ccfg_b(1, 3, 1080, 1920, 2);		// bicubic
	//Configs ccfg_b(1, 3, 256, 256, 2);			// bicubic
	Configs ccfg_c1(ccfg_b);					// convert-1
	Configs ccfg_s(ccfg_c1, 3, 1);				// Split

	// host�������� ����� ������ ���� �Ҵ�
	uint8_t *input = (uint8_t*)malloc(sizeof(uint8_t)* ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W);	// input data [N,C,H,W]
	float *output_b = (float*)malloc(sizeof(float)* ccfg_b.N * ccfg_b.K * ccfg_b.P * ccfg_b.Q);		// bicubic Ouput [N, C, P, Q]
	float *output_c1 = (float*)malloc(sizeof(float)* ccfg_c1.N * ccfg_c1.K * ccfg_c1.P * ccfg_c1.Q);	// convert-1 Ouput [N, C, P, Q]
	float *output_s = (float*)malloc(sizeof(float)* ccfg_s.N * ccfg_s.K * ccfg_s.P * ccfg_s.Q);		// split Ouput [N, C, P, Q]
	//================================================================================================================

	// input data �ʱ�ȭ
	initDataRandom255(input, ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W);
	//================================================================================================================

	// �÷���, ����̽�, ���ؽ�Ʈ, Ŀ�ǵ� ť ���� �κ� (openCL �ڵ忡�� ���� �κ�)
	ocl_args_d_t ocl;
	cl_int status;
	size_t source_size = 0;
	char* clKernelSource = readSource("Template.cl", source_size); // OpenCL Ŀ�� �ҽ� ���� �ε�
	cl_setting(ocl, status, clKernelSource, source_size, true, false);
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// device���� ����� �޸� ���� �Ҵ�
	cl_mem inputBuffer_b = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(uint8_t) * ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W, NULL, &status);	// Input data array on the device
	cl_mem outputBuffer_b = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(float) * ccfg_b.N * ccfg_b.K * ccfg_b.P * ccfg_b.Q, NULL, &status);	// Input data array on the device
	
	//cl_mem inputBuffer_c1 = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(float) * ccfg_c1.N * ccfg_c1.C * ccfg_c1.H * ccfg_c1.W, NULL, &status);	// Input data array on the device
	cl_mem outputBuffer_c1 = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float) * ccfg_c1.N * ccfg_c1.K * ccfg_c1.P * ccfg_c1.Q, NULL, &status);	// Output array on the device
	
	//cl_mem inputBuffer_s = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(float) * ccfg_s.N * ccfg_s.C * ccfg_s.H * ccfg_s.W, NULL, &status);	// Input data array on the device
	cl_mem outputBuffer_s = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float) * ccfg_s.N * ccfg_s.K * ccfg_s.P * ccfg_s.Q, NULL, &status);	// Output array on the device

	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// host�����͸� device�� ���� (host -> device)
	status = clEnqueueWriteBuffer(ocl.commandQueue, inputBuffer_b, CL_FALSE, 0, sizeof(uint8_t) * ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W, input, 0, NULL, NULL);
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	bool time = true;

	for (int i = 0; i < 10000; i++) {
		bicubic_opencl(outputBuffer_b, inputBuffer_b, ccfg_b, ocl, status, time);
		convert1_opencl(outputBuffer_c1, outputBuffer_b, ccfg_c1, ocl, status, time);
		split_opencl(outputBuffer_s, outputBuffer_c1, ccfg_s, ocl, status, time);
	}
	
	// device�� ��� ����� host�� �ҷ����� (device -> host)
	//status = clEnqueueReadBuffer(ocl.commandQueue, outputBuffer_b, CL_TRUE, 0, sizeof(float) * ccfg_b.N * ccfg_b.K * ccfg_b.P * ccfg_b.Q, output, 0, NULL, NULL); // device (bufferC) -> host (C) ����
	//status = clEnqueueReadBuffer(ocl.commandQueue, outputBuffer_c1, CL_TRUE, 0, sizeof(float) * ccfg_c1.N * ccfg_c1.K * ccfg_c1.P * ccfg_c1.Q, output_c1, 0, NULL, NULL); // device (bufferC) -> host (C) ����
	status = clEnqueueReadBuffer(ocl.commandQueue, outputBuffer_s, CL_TRUE, 0, sizeof(float) * ccfg_s.N * ccfg_s.K * ccfg_s.P * ccfg_s.Q, output_s, 0, NULL, NULL); // device (bufferC) -> host (C) ����
	if (status != 0) checkError(status, __LINE__);

	// valication with python results
	/*
	if (0) {// ����
		//printData(output, N, C, P, Q);
		tofile(input, ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W, "../Validation_py/Input_I");
		tofile(output, ccfg_b.N * ccfg_b.K * ccfg_b.P * ccfg_b.Q, "../Validation_py/Output_I");

		// python ���� ��ũ��Ʈ ���� ��ɾ� ����
		printf("\n *Validation with python \n");
		std::string command = "python ../Validation_py/integration.py --N=" + std::to_string(ccfg_b.N) + " --C=" + std::to_string(ccfg_b.C) + " --H=" + std::to_string(ccfg_b.H) + " --W=" + std::to_string(ccfg_b.W);
		const char *cmd = command.c_str();
		// �͹̳ο� ��ɾ� ���� 
		system(cmd);
	}
	*/
	// �ڿ� ����
	// Free OpenCL resources
	//clReleaseKernel(bicubic_kernel);
	clReleaseMemObject(inputBuffer_b);
	clReleaseMemObject(outputBuffer_b);

	// Free host resources
	free(input);
	free(output_b);
	free(output_c1);
	free(output_s);

	return 0;
}