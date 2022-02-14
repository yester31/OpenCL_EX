// This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// OpenCL includes
#include <CL/cl.h>

// Custom header file includes 
#include "utils_cl.h"
#include "layers.h"

int main() {
	// Bicubic Interpolation
	// Input [N, C, H, W] -> Ouput [N, C, P, Q]
	Configs ccfg_b(1, 3, 1080, 1920, 2);				// bicubic
	//Configs ccfg_b(1, 3, 256, 256, 2);					// bicubic
	//Configs ccfg_b(1, 3, 512, 512, 2);				// bicubic
	//Configs ccfg_b(1, 3, 1024, 1024, 2);				// bicubic
	Configs ccfg_c1(ccfg_b);							// convert-1
	Configs ccfg_s(ccfg_c1, 3, 1);						// Split
	Configs ccfg1(ccfg_s, 64, 9, 9, 1, 1, 4, 4, 4, 4);	// conv2d 1
	//Configs ccfg1(ccfg_s, 32, 3, 3, 1, 1, 1, 1, 1, 1);	// conv2d 1
	Configs ccfg2(ccfg1, 32, 5, 5, 1, 1, 2, 2, 2, 2);	// conv2d 2
	//Configs ccfg2(ccfg1, 32, 3, 3, 1, 1, 1, 1, 1, 1);	// conv2d 2
	Configs ccfg3(ccfg2, 1, 5, 5, 1, 1, 2, 2, 2, 2);	// conv2d 3
	//Configs ccfg3(ccfg2, 1, 3, 3, 1, 1, 1, 1, 1, 1);	// conv2d 3
	Configs ccfg_cm(ccfg3, 1, 3);						// combine
	Configs ccfg_c2(ccfg_cm);							// convert-2

	// host공간에서 사용할 데이터 공간 할당
	uint8_t *input	= (uint8_t*)malloc(sizeof(uint8_t)* ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W);	// input data [N,C,H,W]
	float *output_b	= (float*)malloc(sizeof(float)* ccfg_b.N * ccfg_b.K * ccfg_b.P * ccfg_b.Q);		// bicubic Ouput [N, K, P, Q]
	float *output_c1= (float*)malloc(sizeof(float)* ccfg_c1.N * ccfg_c1.K * ccfg_c1.P * ccfg_c1.Q);	// convert-1 Ouput [N, K, P, Q]
	float *output_s	= (float*)malloc(sizeof(float)* ccfg_s.N * ccfg_s.K * ccfg_s.P * ccfg_s.Q);		// split Ouput [N, K, P, Q]

	float *weight1 = (float*)malloc(sizeof(float) * ccfg1.K * ccfg1.C * ccfg1.KH * ccfg1.KW);		// conv2d-1 weight [K,C,KH,KW]
	float *output1 = (float*)calloc(ccfg1.K * ccfg1.P * ccfg1.Q * ccfg1.N, sizeof(float));			// conv2d-1 output [N, K, P, Q]
	float *bias1   = (float*)calloc(ccfg1.K, sizeof(float));										// conv2d-1 bias [K]
	float *weight2 = (float*)malloc(sizeof(float) * ccfg2.K * ccfg2.C * ccfg2.KH * ccfg2.KW);		// conv2d-2 weight [K,C,KH,KW]
	float *output2 = (float*)calloc(ccfg2.K * ccfg2.P * ccfg2.Q * ccfg2.N, sizeof(float));			// conv2d-2 output [N, K, P, Q]
	float *bias2 = (float*)calloc(ccfg2.K, sizeof(float));											// conv2d-2 bias [K]
	float *weight3 = (float*)malloc(sizeof(float) * ccfg3.K * ccfg3.C * ccfg3.KH * ccfg3.KW);		// conv2d-3 weight [K,C,KH,KW]
	float *output3 = (float*)calloc(ccfg3.K * ccfg3.P * ccfg3.Q * ccfg3.N, sizeof(float));			// conv2d-3 output [N, K, P, Q]
	float *bias3 = (float*)calloc(ccfg3.K, sizeof(float));											// conv2d-3 bias [K]

	float *output_cm = (float*)malloc(sizeof(float)* ccfg_cm.N * ccfg_cm.K * ccfg_cm.P * ccfg_cm.Q);	// combine Ouput [N, K, P, Q]
	float *output_c2 = (float*)malloc(sizeof(float)* ccfg_c2.N * ccfg_c2.K * ccfg_c2.P * ccfg_c2.Q);	// convert-2 Ouput [N, K, P, Q]
	//================================================================================================================

	// input data 초기화
	initDataRandom255(input, ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W);
	initDataRandomZP1(weight1, ccfg1.K * ccfg1.C * ccfg1.KH * ccfg1.KW);
	initDataRandomZP1(bias1, ccfg1.K);
	initDataRandomZP1(weight2, ccfg2.K * ccfg2.C * ccfg2.KH * ccfg2.KW);
	initDataRandomZP1(bias2, ccfg2.K);
	initDataRandomZP1(weight3, ccfg3.K * ccfg3.C * ccfg3.KH * ccfg3.KW);
	initDataRandomZP1(bias3, ccfg3.K);
	//================================================================================================================

	// 플랫폼, 디바이스, 컨텍스트, 커맨드 큐 설정 부분 (openCL 코드에서 공통 부분)
	ocl_args_d_t ocl;
	cl_int status;
	size_t source_size = 0;
	char* clKernelSource = readSource("Template.cl", source_size); // OpenCL 커널 소스 파일 로드
	cl_setting(ocl, status, clKernelSource, source_size, true, false);

	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// device에서 사용할 메모리 공간 할당
	cl_mem inputBuffer_b	= clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(uint8_t) * ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W, NULL, &status);	// Input data array on the device

	cl_mem outputBuffer_b	= clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg_b.N	* ccfg_b.K	* ccfg_b.P	* ccfg_b.Q,		NULL, &status);	// Input data array on the device
	cl_mem outputBuffer_c1	= clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg_c1.N	* ccfg_c1.K	* ccfg_c1.P	* ccfg_c1.Q,	NULL, &status);	// Output array on the device
	cl_mem outputBuffer_s	= clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg_s.N	* ccfg_s.K	* ccfg_s.P	* ccfg_s.Q,		NULL, &status);	// Output array on the device
	cl_mem outputBuffer1	= clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg1.N	* ccfg1.K	* ccfg1.P	* ccfg1.Q,		NULL, &status);	// Output array on the device
	cl_mem outputBuffer2	= clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg2.N	* ccfg2.K	* ccfg2.P	* ccfg2.Q,		NULL, &status);	// Output array on the device
	cl_mem outputBuffer3	= clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg3.N	* ccfg3.K	* ccfg3.P	* ccfg3.Q,		NULL, &status);	// Output array on the device
	cl_mem outputBuffer_cm	= clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg_cm.N	* ccfg_cm.K * ccfg_cm.P	* ccfg_cm.Q,	NULL, &status);	// Output array on the device
	cl_mem outputBuffer_c2	= clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg_c2.N	* ccfg_c2.K * ccfg_c2.P	* ccfg_c2.Q,	NULL, &status);	// Output array on the device

	cl_mem weightBuffer1	= clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(float) * ccfg1.K * ccfg1.C * ccfg1.KH * ccfg1.KW, NULL, &status);// Input weight array on the device
	cl_mem biasBuffer1	= clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(float) * ccfg1.K, NULL, &status);// Input weight array on the device
	cl_mem weightBuffer2	= clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(float) * ccfg2.K * ccfg2.C * ccfg2.KH * ccfg2.KW, NULL, &status);// Input weight array on the device
	cl_mem biasBuffer2	= clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(float) * ccfg2.K, NULL, &status);// Input weight array on the device
	cl_mem weightBuffer3	= clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(float) * ccfg3.K * ccfg3.C * ccfg3.KH * ccfg3.KW, NULL, &status);// Input weight array on the device
	cl_mem biasBuffer3	= clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(float) * ccfg3.K, NULL, &status);// Input weight array on the device

	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// host데이터를 device로 전달 (host -> device)
	status = clEnqueueWriteBuffer(ocl.commandQueue, inputBuffer_b, CL_FALSE, 0, sizeof(uint8_t) * ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W, input, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(ocl.commandQueue, weightBuffer1, CL_FALSE, 0, sizeof(float) * ccfg1.K * ccfg1.C * ccfg1.KH* ccfg1.KW, weight1, 0, NULL, NULL); // host (weight1) -> device (buffer1)전달
	status = clEnqueueWriteBuffer(ocl.commandQueue, biasBuffer1, CL_FALSE, 0, sizeof(float) * ccfg1.K , bias1, 0, NULL, NULL); // host (weight1) -> device (buffer1)전달
	status = clEnqueueWriteBuffer(ocl.commandQueue, weightBuffer2, CL_FALSE, 0, sizeof(float) * ccfg2.K * ccfg2.C * ccfg2.KH* ccfg2.KW, weight2, 0, NULL, NULL); // host (weight1) -> device (buffer1)전달
	status = clEnqueueWriteBuffer(ocl.commandQueue, biasBuffer2, CL_FALSE, 0, sizeof(float) * ccfg2.K, bias2, 0, NULL, NULL); // host (weight1) -> device (buffer1)전달
	status = clEnqueueWriteBuffer(ocl.commandQueue, weightBuffer3, CL_FALSE, 0, sizeof(float) * ccfg3.K * ccfg3.C * ccfg3.KH* ccfg3.KW, weight3, 0, NULL, NULL); // host (weight1) -> device (buffer1)전달
	status = clEnqueueWriteBuffer(ocl.commandQueue, biasBuffer3, CL_FALSE, 0, sizeof(float) * ccfg3.K , bias3, 0, NULL, NULL); // host (weight1) -> device (buffer1)전달

	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	bool time = true;

	for (int i = 0; i < 1000; i++) {
		uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

		//cl_mem outputBuffer_b = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg_b.N * ccfg_b.K * ccfg_b.P * ccfg_b.Q, NULL, &status);	// Input data array on the device
		bicubic_opencl(outputBuffer_b, inputBuffer_b, ccfg_b, ocl, status, time);

		//cl_mem outputBuffer_c1 = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg_c1.N * ccfg_c1.K * ccfg_c1.P * ccfg_c1.Q, NULL, &status);	// Output array on the device
		convert1_opencl(outputBuffer_c1, outputBuffer_b, ccfg_c1, ocl, status, time);

		//cl_mem outputBuffer_s = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg_s.N * ccfg_s.K * ccfg_s.P * ccfg_s.Q, NULL, &status);	// Output array on the device
		split_opencl(outputBuffer_s, outputBuffer_c1, ccfg_s, ocl, status, time);

		//cl_mem outputBuffer1 = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float)* ccfg1.N * ccfg1.K * ccfg1.P * ccfg1.Q, NULL, &status);	// Output array on the device
		conv2d_opencl(outputBuffer1, outputBuffer_s, weightBuffer1, biasBuffer1, ccfg1, ocl, status, time);

		//cl_mem outputBuffer2 = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float)* ccfg2.N * ccfg2.K * ccfg2.P * ccfg2.Q, NULL, &status);	// Output array on the device
		conv2d_opencl(outputBuffer2, outputBuffer1, weightBuffer2, biasBuffer2, ccfg2, ocl, status, time);

		//cl_mem outputBuffer3 = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float)* ccfg3.N * ccfg3.K * ccfg3.P * ccfg3.Q, NULL, &status);	// Output array on the device
		conv2d_opencl(outputBuffer3, outputBuffer2, weightBuffer3, biasBuffer3, ccfg3, ocl, status, time);

		//cl_mem outputBuffer_c1 = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float)* ccfg_cm.N * ccfg_cm.K * ccfg_cm.P * ccfg_cm.Q, NULL, &status);	// Output array on the device
		combine_opencl(outputBuffer_c1, outputBuffer3, ccfg_cm, ocl, status, time);

		convert2_opencl(outputBuffer_c2, outputBuffer_c1, ccfg_cm, ocl, status, time);

		uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		printf("\n[total]     dur_time(opencl)            = %6.5f [msec], FPS = %6.2f \n", (start_time2 - start_time1) / 1000.f, (1000000.f / (start_time2 - start_time1)));
		printf("========================================================================\n");
	}

	// device의 계산 결과를 host로 불러오기 (device -> host)
	//status = clEnqueueReadBuffer(ocl.commandQueue, outputBuffer_b, CL_TRUE, 0, sizeof(float) * ccfg_b.N * ccfg_b.K * ccfg_b.P * ccfg_b.Q, output, 0, NULL, NULL); // device (bufferC) -> host (C) 전달
	//status = clEnqueueReadBuffer(ocl.commandQueue, outputBuffer_c1, CL_TRUE, 0, sizeof(float) * ccfg_c1.N * ccfg_c1.K * ccfg_c1.P * ccfg_c1.Q, output_c1, 0, NULL, NULL); // device (bufferC) -> host (C) 전달
	//status = clEnqueueReadBuffer(ocl.commandQueue, outputBuffer_s, CL_TRUE, 0, sizeof(float) * ccfg_s.N * ccfg_s.K * ccfg_s.P * ccfg_s.Q, output_s, 0, NULL, NULL); // device (bufferC) -> host (C) 전달
	//status = clEnqueueReadBuffer(ocl.commandQueue, outputBuffer1, CL_TRUE, 0, sizeof(float) * ccfg1.N * ccfg1.K * ccfg1.P * ccfg1.Q, output1, 0, NULL, NULL); // device (bufferC) -> host (C) 전달
	//status = clEnqueueReadBuffer(ocl.commandQueue, outputBuffer2, CL_TRUE, 0, sizeof(float) * ccfg2.N * ccfg2.K * ccfg2.P * ccfg2.Q, output2, 0, NULL, NULL); // device (bufferC) -> host (C) 전달
	//status = clEnqueueReadBuffer(ocl.commandQueue, outputBuffer3, CL_TRUE, 0, sizeof(float) * ccfg3.N * ccfg3.K * ccfg3.P * ccfg3.Q, output3, 0, NULL, NULL); // device (bufferC) -> host (C) 전달
	//status = clEnqueueReadBuffer(ocl.commandQueue, outputBuffer_c1, CL_TRUE, 0, sizeof(float) * ccfg_cm.N * ccfg_cm.K * ccfg_cm.P * ccfg_cm.Q, output_cm, 0, NULL, NULL); // device (bufferC) -> host (C) 전달
	status = clEnqueueReadBuffer(ocl.commandQueue, outputBuffer_c2, CL_TRUE, 0, sizeof(float) * ccfg_c2.N * ccfg_c2.K * ccfg_c2.P * ccfg_c2.Q, output_c2, 0, NULL, NULL); // device (bufferC) -> host (C) 전달

	if (status != 0) checkError(status, __LINE__);

	// valication with python results
	/*
	if (0) {// 검증
		//printData(output, N, C, P, Q);
		tofile(input, ccfg_b.N * ccfg_b.C * ccfg_b.H * ccfg_b.W, "../Validation_py/Input_I");
		tofile(output, ccfg_b.N * ccfg_b.K * ccfg_b.P * ccfg_b.Q, "../Validation_py/Output_I");

		// python 검증 스크립트 수행 명령어 세팅
		printf("\n *Validation with python \n");
		std::string command = "python ../Validation_py/integration.py --N=" + std::to_string(ccfg_b.N) + " --C=" + std::to_string(ccfg_b.C) + " --H=" + std::to_string(ccfg_b.H) + " --W=" + std::to_string(ccfg_b.W);
		const char *cmd = command.c_str();
		// 터미널에 명령어 전달
		system(cmd);
	}
	*/


	// 자원 해제
	// Free OpenCL resources
	clReleaseMemObject(inputBuffer_b);
	clReleaseMemObject(outputBuffer_b);
	clReleaseMemObject(outputBuffer_c1);
	clReleaseMemObject(outputBuffer_s);
	clReleaseMemObject(weightBuffer1);
	clReleaseMemObject(weightBuffer2);
	clReleaseMemObject(weightBuffer3);
	clReleaseMemObject(biasBuffer1);
	clReleaseMemObject(biasBuffer2);
	clReleaseMemObject(biasBuffer3);
	clReleaseMemObject(outputBuffer1);
	clReleaseMemObject(outputBuffer2);
	clReleaseMemObject(outputBuffer3);
	clReleaseMemObject(outputBuffer_c2);

	// Free host resources
	free(input);
	free(output_b);
	free(output_c1);
	free(output_s);
	free(output1);
	free(output2);
	free(output3);
	free(weight1);
	free(weight2);
	free(weight3);
	free(bias1);
	free(bias2);
	free(bias3);
	free(output_cm);
	free(output_c2);

	return 0;
}