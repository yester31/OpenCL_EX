#pragma once

void bicubic_opencl(cl_mem outputBuffer, cl_mem inputBuffer, Configs &ccfg, ocl_args_d_t &ocl, cl_int status, bool time = false)
{
	//printf("[bicubic]   input [%4d,%4d,%4d,%4d] -> output [%4d,%4d,%4d,%4d]  \n", ccfg.N, ccfg.C, ccfg.H, ccfg.W, ccfg.N, ccfg.K, ccfg.P, ccfg.Q);
	//status = clFinish(ocl.commandQueue);
	//if (status != 0) checkError(status, __LINE__);

	// 커널 생성 (커널 함수 이름을 인자로 전달)
	cl_kernel bicubic_kernel = clCreateKernel(ocl.program, "bicubic_kernel", &status); 
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// 커널에서 사용할 인자 값 계산 작업
	float h_scale = float(ccfg.H - 1) / (ccfg.P - 1);
	float w_scale = float(ccfg.W - 1) / (ccfg.Q - 1);
	int num_elements = ccfg.N * ccfg.K * ccfg.P * ccfg.Q;
	//================================================================================================================

	// 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(bicubic_kernel, 0, sizeof(cl_mem), &outputBuffer); // output
	status = clSetKernelArg(bicubic_kernel, 1, sizeof(cl_mem), &inputBuffer); // input
	status = clSetKernelArg(bicubic_kernel, 2, sizeof(cl_float), &h_scale); 
	status = clSetKernelArg(bicubic_kernel, 3, sizeof(cl_float), &w_scale); 
	status = clSetKernelArg(bicubic_kernel, 4, sizeof(cl_int), &ccfg.N); 
	status = clSetKernelArg(bicubic_kernel, 5, sizeof(cl_int), &ccfg.C); 
	status = clSetKernelArg(bicubic_kernel, 6, sizeof(cl_int), &ccfg.H);
	status = clSetKernelArg(bicubic_kernel, 7, sizeof(cl_int), &ccfg.W); 
	status = clSetKernelArg(bicubic_kernel, 8, sizeof(cl_int), &ccfg.P); 
	status = clSetKernelArg(bicubic_kernel, 9, sizeof(cl_int), &ccfg.Q); 
	status = clSetKernelArg(bicubic_kernel, 10, sizeof(cl_int), &num_elements); 
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// 실행을 위한 워크 아이템의 인덱스 공간(글로벌 워크 사이즈) 정의
	size_t globalWorkSize[1]; 
	globalWorkSize[0] = num_elements;
	//================================================================================================================

	if (time) {
		// 커널 실행, event를 이용한 커널 실행 시간 측정
		cl_event event;
		status = clEnqueueNDRangeKernel(ocl.commandQueue, bicubic_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event);
		if (status != 0) checkError(status, __LINE__);
		status = clWaitForEvents(1, &event);
		if (status != 0) checkError(status, __LINE__);
		status = clFinish(ocl.commandQueue);
		if (status != 0) checkError(status, __LINE__);
		//clReleaseMemObject(inputBuffer);
		cl_ulong time_start;
		cl_ulong time_end;
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		printf("[bicubic]   dur_time(openCL)           = %6.5f [msec] \n", (time_end - time_start) / 1000000.0);
		//================================================================================================================
		//clReleaseKernel(bicubic_kernel);
	}
	else {
		// 커널 실행
		status = clEnqueueNDRangeKernel(ocl.commandQueue, bicubic_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
		if (status != 0) checkError(status, __LINE__);
		status = clFinish(ocl.commandQueue);
		if (status != 0) checkError(status, __LINE__);
		//clReleaseMemObject(inputBuffer);
		//================================================================================================================
		//clReleaseKernel(bicubic_kernel);
	}
}


void convert1_opencl(cl_mem outputBuffer, cl_mem inputBuffer, Configs &ccfg, ocl_args_d_t &ocl, cl_int status, bool time = false)
{
	//printf("[convert1]  input [%4d,%4d,%4d,%4d] -> output [%4d,%4d,%4d,%4d]  \n", ccfg.N, ccfg.C, ccfg.H, ccfg.W, ccfg.N, ccfg.K, ccfg.P, ccfg.Q);
	//status = clFinish(ocl.commandQueue);
	//if (status != 0) checkError(status, __LINE__);
	// 커널 생성 (커널 함수 이름을 인자로 전달)
	cl_kernel convert1_kernel = clCreateKernel(ocl.program, "convert1_kernel", &status);
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// 커널에서 사용할 인자 값 계산 작업
	int num_elements = ccfg.N * ccfg.P * ccfg.Q;
	//================================================================================================================

	// 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(convert1_kernel, 0, sizeof(cl_mem), &outputBuffer); // output
	status = clSetKernelArg(convert1_kernel, 1, sizeof(cl_mem), &inputBuffer); // input
	status = clSetKernelArg(convert1_kernel, 2, sizeof(cl_int), &ccfg.N);
	status = clSetKernelArg(convert1_kernel, 3, sizeof(cl_int), &ccfg.C);
	status = clSetKernelArg(convert1_kernel, 4, sizeof(cl_int), &ccfg.H);
	status = clSetKernelArg(convert1_kernel, 5, sizeof(cl_int), &ccfg.W);
	status = clSetKernelArg(convert1_kernel, 6, sizeof(cl_int), &num_elements);
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// 실행을 위한 워크 아이템의 인덱스 공간(글로벌 워크 사이즈) 정의
	size_t globalWorkSize[1];
	globalWorkSize[0] = num_elements;
	//================================================================================================================

	if (time) {
		// 커널 실행, event를 이용한 커널 실행 시간 측정
		cl_event event;
		status = clEnqueueNDRangeKernel(ocl.commandQueue, convert1_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event);
		if (status != 0) checkError(status, __LINE__);
		status = clWaitForEvents(1, &event);
		if (status != 0) checkError(status, __LINE__);
		status = clFinish(ocl.commandQueue);
		if (status != 0) checkError(status, __LINE__);
		//clReleaseMemObject(inputBuffer);
		cl_ulong time_start;
		cl_ulong time_end;
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		printf("[convert1]  dur_time(openCL)           = %6.5f [msec] \n", (time_end - time_start) / 1000000.0);
		//================================================================================================================
		//clReleaseKernel(convert1_kernel);
	}
	else {
		// 커널 실행
		status = clEnqueueNDRangeKernel(ocl.commandQueue, convert1_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
		if (status != 0) checkError(status, __LINE__);
		status = clFinish(ocl.commandQueue);
		if (status != 0) checkError(status, __LINE__);
		//clReleaseMemObject(inputBuffer);
		//================================================================================================================
		//clReleaseKernel(convert1_kernel);
	}
}

void split_opencl(cl_mem outputBuffer, cl_mem inputBuffer, Configs &ccfg, ocl_args_d_t &ocl, cl_int status, bool time = false)
{
	//printf("[split]     input [%4d,%4d,%4d,%4d] -> output [%4d,%4d,%4d,%4d]  \n", ccfg.N, ccfg.C, ccfg.H, ccfg.W, ccfg.N, ccfg.K, ccfg.P, ccfg.Q);
	//status = clFinish(ocl.commandQueue);
	//if (status != 0) checkError(status, __LINE__);
	if (1) {
		// 커널 생성 (커널 함수 이름을 인자로 전달)
		cl_kernel split_kernel = clCreateKernel(ocl.program, "split_first_channel_kernel", &status);
		if (status != 0) checkError(status, __LINE__);
		//================================================================================================================

		// 커널에서 사용할 인자 값 계산 작업
		int num_elements = ccfg.N * ccfg.K * ccfg.P * ccfg.Q;
		//================================================================================================================

		// 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(split_kernel, 0, sizeof(cl_mem), &outputBuffer);	// output
		status = clSetKernelArg(split_kernel, 1, sizeof(cl_mem), &inputBuffer);		// input
		status = clSetKernelArg(split_kernel, 2, sizeof(cl_int), &num_elements);
		if (status != 0) checkError(status, __LINE__);
		//================================================================================================================

		// 실행을 위한 워크 아이템의 인덱스 공간(글로벌 워크 사이즈) 정의
		size_t globalWorkSize[1];
		globalWorkSize[0] = num_elements;
		//================================================================================================================

		if (time) {
			// 커널 실행, event를 이용한 커널 실행 시간 측정
			cl_event event;
			status = clEnqueueNDRangeKernel(ocl.commandQueue, split_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event);
			if (status != 0) checkError(status, __LINE__);
			status = clWaitForEvents(1, &event);
			if (status != 0) checkError(status, __LINE__);
			status = clFinish(ocl.commandQueue);
			if (status != 0) checkError(status, __LINE__);
			//clReleaseMemObject(inputBuffer);
			cl_ulong time_start;
			cl_ulong time_end;
			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
			printf("[split]     dur_time(openCL)           = %6.5f [msec] \n", (time_end - time_start) / 1000000.0);
			//================================================================================================================
			//clReleaseKernel(split_kernel);
		}
		else {
			// 커널 실행
			status = clEnqueueNDRangeKernel(ocl.commandQueue, split_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
			if (status != 0) checkError(status, __LINE__);
			status = clFinish(ocl.commandQueue);
			if (status != 0) checkError(status, __LINE__);
			//clReleaseMemObject(inputBuffer);
			//================================================================================================================
			//clReleaseKernel(split_kernel);
		}
	}
	else {
		// create buffer for device 
		cl_mem output2 = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg.N * (ccfg.C - ccfg.K) * ccfg.P * ccfg.Q, NULL, &status);	// Output im2col array on the device
		if (status != 0) checkError(status, __LINE__);
	
		// 커널 생성 (커널 함수 이름을 인자로 전달)
		cl_kernel split_kernel = clCreateKernel(ocl.program, "split_kernel", &status);
		if (status != 0) checkError(status, __LINE__);
		//================================================================================================================

		// 커널에서 사용할 인자 값 계산 작업
		int num_elements = ccfg.N * ccfg.K * ccfg.P * ccfg.Q;
		//================================================================================================================

		// 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(split_kernel, 0, sizeof(cl_mem), &outputBuffer);	// output
		status = clSetKernelArg(split_kernel, 1, sizeof(cl_mem), &output2);			// output2
		status = clSetKernelArg(split_kernel, 2, sizeof(cl_mem), &inputBuffer);		// input
		status = clSetKernelArg(split_kernel, 3, sizeof(cl_int), &ccfg.N);
		status = clSetKernelArg(split_kernel, 4, sizeof(cl_int), &ccfg.C);
		status = clSetKernelArg(split_kernel, 5, sizeof(cl_int), &ccfg.H);
		status = clSetKernelArg(split_kernel, 6, sizeof(cl_int), &ccfg.W);
		status = clSetKernelArg(split_kernel, 7, sizeof(cl_int), &num_elements);
		if (status != 0) checkError(status, __LINE__);
		//================================================================================================================

		// 실행을 위한 워크 아이템의 인덱스 공간(글로벌 워크 사이즈) 정의
		size_t globalWorkSize[1];
		globalWorkSize[0] = num_elements;
		//================================================================================================================

		if (time) {
			// 커널 실행, event를 이용한 커널 실행 시간 측정
			cl_event event;
			status = clEnqueueNDRangeKernel(ocl.commandQueue, split_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event);
			if (status != 0) checkError(status, __LINE__);
			status = clWaitForEvents(1, &event);
			if (status != 0) checkError(status, __LINE__);
			status = clFinish(ocl.commandQueue);
			if (status != 0) checkError(status, __LINE__);
			//clReleaseMemObject(inputBuffer);
			cl_ulong time_start;
			cl_ulong time_end;
			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
			printf("[split]     dur_time(openCL)           = %6.5f [msec] \n", (time_end - time_start) / 1000000.0);
			//================================================================================================================
		}
		else {
			// 커널 실행
			status = clEnqueueNDRangeKernel(ocl.commandQueue, split_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
			if (status != 0) checkError(status, __LINE__);
			status = clFinish(ocl.commandQueue);
			if (status != 0) checkError(status, __LINE__);
			//clReleaseMemObject(inputBuffer);
			//================================================================================================================
		}
		clReleaseMemObject(output2);
	}
}

void conv2d_opencl(cl_mem outputBuffer, cl_mem inputBuffer, cl_mem weightBuffer, cl_mem biasBuffer, Configs &ccfg, ocl_args_d_t &ocl, cl_int status, bool time = false)
{
	//printf("[conv2d]    input [%4d,%4d,%4d,%4d] -> output [%4d,%4d,%4d,%4d]  \n", ccfg.N, ccfg.C, ccfg.H, ccfg.W, ccfg.N, ccfg.K, ccfg.P, ccfg.Q);
	//status = clFinish(ocl.commandQueue);
	//if (status != 0) checkError(status, __LINE__);
	if (1) {
		// 커널 생성 (커널 함수 이름을 인자로 전달)
		cl_kernel conv2d_kernel = clCreateKernel(ocl.program, "conv2d_kernel", &status);
		if (status != 0) checkError(status, __LINE__);
		//================================================================================================================

		// 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 0, sizeof(cl_mem), &outputBuffer); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 1, sizeof(cl_mem), &inputBuffer); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 2, sizeof(cl_mem), &weightBuffer); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 3, sizeof(cl_mem), &biasBuffer); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 4, sizeof(cl_int), &ccfg.N); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 5, sizeof(cl_int), &ccfg.C); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 6, sizeof(cl_int), &ccfg.H); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 7, sizeof(cl_int), &ccfg.W); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 8, sizeof(cl_int), &ccfg.K); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 9, sizeof(cl_int), &ccfg.P); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 10, sizeof(cl_int), &ccfg.Q); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 11, sizeof(cl_int), &ccfg.KH); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 12, sizeof(cl_int), &ccfg.KW); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 13, sizeof(cl_int), &ccfg.SH); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 14, sizeof(cl_int), &ccfg.SW); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 15, sizeof(cl_int), &ccfg.PL); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		status = clSetKernelArg(conv2d_kernel, 16, sizeof(cl_int), &ccfg.PT); // 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		if (status != 0) checkError(status, __LINE__);


		//================================================================================================================

		// 실행을 위한 워크 아이템의 인덱스 공간(글로벌 워크 사이즈) 정의
		size_t globalWorkSize[1] = { ccfg.N * ccfg.K * ccfg.P * ccfg.Q };

		//================================================================================================================

		if (time) {
			// 커널 실행, event를 이용한 커널 실행 시간 측정
			cl_event event;
			status = clEnqueueNDRangeKernel(ocl.commandQueue, conv2d_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event);
			if (status != 0) checkError(status, __LINE__);
			status = clWaitForEvents(1, &event);
			if (status != 0) checkError(status, __LINE__);
			status = clFinish(ocl.commandQueue);
			if (status != 0) checkError(status, __LINE__);
			//clReleaseMemObject(inputBuffer);
			cl_ulong time_start;
			cl_ulong time_end;
			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
			printf("[conv2d]    dur_time(openCL)           = %6.5f [msec] \n", (time_end - time_start) / 1000000.0);
			//================================================================================================================
		}
		else {
			// 커널 실행
			// executes conv2d_kernel
			status = clEnqueueNDRangeKernel(ocl.commandQueue, conv2d_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
			if (status != 0) checkError(status, __LINE__);
			status = clFinish(ocl.commandQueue);
			if (status != 0) checkError(status, __LINE__);
			//================================================================================================================
		}
	}else{
		// 임시 메모리 공간 할당
		cl_mem im2colBuffer = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg.C * ccfg.KH * ccfg.KW * ccfg.N * ccfg.P * ccfg.Q, NULL, &status);
		cl_mem middleBuffer = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(float) * ccfg.K * ccfg.P * ccfg.Q * ccfg.N, NULL, &status);
		if (status != 0) checkError(status, __LINE__);
		status = clFinish(ocl.commandQueue);
		if (status != 0) checkError(status, __LINE__);
		//================================================================================================================
	
		// 커널 생성 (커널 함수 이름을 인자로 전달)
		cl_kernel im2col_kernel = clCreateKernel(ocl.program, "im2col_kernel", &status);
		cl_kernel matMul_kernel = clCreateKernel(ocl.program, "matMul_kernel", &status);
		cl_kernel col2im_kernel = clCreateKernel(ocl.program, "col2im_kernel", &status);
		if (status != 0) checkError(status, __LINE__);
		//================================================================================================================
	
		// 커널에서 사용할 인자 값 계산 작업
		int k_v = ccfg.C * ccfg.KH* ccfg.KW;
		int n_v = ccfg.P * ccfg.Q * ccfg.N;
		//================================================================================================================

		// 커널 함수 인자 값 전달 (버퍼와 커널 연결)
		// set im2col_kernel arguments for kernel
		status = clSetKernelArg(im2col_kernel, 0, sizeof(cl_mem), &im2colBuffer);
		status = clSetKernelArg(im2col_kernel, 1, sizeof(cl_mem), &inputBuffer);
		status = clSetKernelArg(im2col_kernel, 2, sizeof(cl_int), &ccfg.N);
		status = clSetKernelArg(im2col_kernel, 3, sizeof(cl_int), &ccfg.K);
		status = clSetKernelArg(im2col_kernel, 4, sizeof(cl_int), &ccfg.P);
		status = clSetKernelArg(im2col_kernel, 5, sizeof(cl_int), &ccfg.Q);
		status = clSetKernelArg(im2col_kernel, 6, sizeof(cl_int), &ccfg.C);
		status = clSetKernelArg(im2col_kernel, 7, sizeof(cl_int), &ccfg.H);
		status = clSetKernelArg(im2col_kernel, 8, sizeof(cl_int), &ccfg.W);
		status = clSetKernelArg(im2col_kernel, 9, sizeof(cl_int), &ccfg.KH);
		status = clSetKernelArg(im2col_kernel, 10, sizeof(cl_int), &ccfg.KW);
		status = clSetKernelArg(im2col_kernel, 11, sizeof(cl_int), &ccfg.SH);
		status = clSetKernelArg(im2col_kernel, 12, sizeof(cl_int), &ccfg.SW);
		status = clSetKernelArg(im2col_kernel, 13, sizeof(cl_int), &ccfg.PL);
		status = clSetKernelArg(im2col_kernel, 14, sizeof(cl_int), &ccfg.PT);
		if (status != 0) checkError(status, __LINE__);

		// set matMul_kernel arguments for kernel
		status = clSetKernelArg(matMul_kernel, 0, sizeof(cl_mem), &weightBuffer);
		status = clSetKernelArg(matMul_kernel, 1, sizeof(cl_mem), &im2colBuffer);
		status = clSetKernelArg(matMul_kernel, 2, sizeof(cl_mem), &middleBuffer);
		status = clSetKernelArg(matMul_kernel, 3, sizeof(cl_int), &ccfg.K);
		status = clSetKernelArg(matMul_kernel, 4, sizeof(cl_int), &n_v);
		status = clSetKernelArg(matMul_kernel, 5, sizeof(cl_int), &k_v);
		if (status != 0) checkError(status, __LINE__);

		// set col2im_kernel arguments for kernel
		status = clSetKernelArg(col2im_kernel, 0, sizeof(cl_mem), &outputBuffer);
		status = clSetKernelArg(col2im_kernel, 1, sizeof(cl_mem), &middleBuffer);
		status = clSetKernelArg(col2im_kernel, 2, sizeof(cl_mem), &biasBuffer);
		status = clSetKernelArg(col2im_kernel, 3, sizeof(cl_int), &ccfg.N);
		status = clSetKernelArg(col2im_kernel, 4, sizeof(cl_int), &ccfg.K);
		status = clSetKernelArg(col2im_kernel, 5, sizeof(cl_int), &ccfg.P);
		status = clSetKernelArg(col2im_kernel, 6, sizeof(cl_int), &ccfg.Q);
		if (status != 0) checkError(status, __LINE__);
		//================================================================================================================

		// 실행을 위한 워크 아이템의 인덱스 공간(글로벌 워크 사이즈) 정의
		size_t globalWorkSize[1] = { ccfg.C * ccfg.KH * ccfg.KW * ccfg.N * ccfg.P * ccfg.Q };
		size_t globalWorkSize2[1] = { ccfg.K * ccfg.N * ccfg.P * ccfg.Q };
		size_t globalWorkSize3[1] = { ccfg.K * ccfg.N * ccfg.P * ccfg.Q };
		//================================================================================================================

		if (time) {
			// 커널 실행, event를 이용한 커널 실행 시간 측정
			cl_event event[3];

			// executes im2col_kernel
			status = clEnqueueNDRangeKernel(ocl.commandQueue, im2col_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event[0]);
			if (status != 0) checkError(status, __LINE__);
			//clReleaseMemObject(inputBuffer);
		
			// executes matMul_kernel
			status = clEnqueueNDRangeKernel(ocl.commandQueue, matMul_kernel, 1, NULL, globalWorkSize2, NULL, 0, NULL, &event[1]);
			if (status != 0) checkError(status, __LINE__);
			//clReleaseMemObject(weightBuffer);
			clReleaseMemObject(im2colBuffer);
		
			// executes col2im_kernel
			status = clEnqueueNDRangeKernel(ocl.commandQueue, col2im_kernel, 1, NULL, globalWorkSize3, NULL, 0, NULL, &event[2]);
			if (status != 0) checkError(status, __LINE__);
			clReleaseMemObject(middleBuffer);

			status = clWaitForEvents(3, event);
			if (status != 0) checkError(status, __LINE__);

			status = clFinish(ocl.commandQueue);
			if (status != 0) checkError(status, __LINE__);
			//clReleaseMemObject(inputBuffer);

			cl_ulong time_start;
			cl_ulong time_end;
			clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
			clGetEventProfilingInfo(event[2], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
			printf("[conv2d]    dur_time(openCL)           = %6.5f [msec] \n", (time_end - time_start) / 1000000.0);
			//================================================================================================================
			//clReleaseKernel(im2col_kernel);
			//clReleaseKernel(col2im_kernel);
			//clReleaseKernel(matMul_kernel);
		}
		else {
			// 커널 실행
			// executes im2col_kernel
			status = clEnqueueNDRangeKernel(ocl.commandQueue, im2col_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
			if (status != 0) checkError(status, __LINE__);
			//clReleaseMemObject(inputBuffer);

			// executes matMul_kernel
			status = clEnqueueNDRangeKernel(ocl.commandQueue, matMul_kernel, 1, NULL, globalWorkSize2, NULL, 0, NULL, NULL);
			if (status != 0) checkError(status, __LINE__);
			//clReleaseMemObject(weightBuffer);
			clReleaseMemObject(im2colBuffer);

			// executes col2im_kernel
			status = clEnqueueNDRangeKernel(ocl.commandQueue, col2im_kernel, 1, NULL, globalWorkSize3, NULL, 0, NULL, NULL);
			if (status != 0) checkError(status, __LINE__);
			clReleaseMemObject(middleBuffer);

			status = clFinish(ocl.commandQueue);
			if (status != 0) checkError(status, __LINE__);
			//================================================================================================================
			//clReleaseKernel(im2col_kernel);
			//clReleaseKernel(col2im_kernel);
			//clReleaseKernel(matMul_kernel);
		}
	}
}

void combine_opencl(cl_mem outputBuffer, cl_mem inputBuffer, Configs &ccfg, ocl_args_d_t &ocl, cl_int status, bool time = false)
{
	//printf("[combine]   input [%4d,%4d,%4d,%4d] -> output [%4d,%4d,%4d,%4d]  \n", ccfg.N, ccfg.C, ccfg.H, ccfg.W, ccfg.N, ccfg.K, ccfg.P, ccfg.Q);
	//status = clFinish(ocl.commandQueue);
	//if (status != 0) checkError(status, __LINE__);
	// 커널 생성 (커널 함수 이름을 인자로 전달)
	cl_kernel combine_kernel = clCreateKernel(ocl.program, "combine_first_channel_kernel", &status);
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// 커널에서 사용할 인자 값 계산 작업
	int num_elements = ccfg.N * ccfg.C * ccfg.H * ccfg.W;
	//================================================================================================================

	// 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(combine_kernel, 0, sizeof(cl_mem), &outputBuffer); // output
	status = clSetKernelArg(combine_kernel, 1, sizeof(cl_mem), &inputBuffer); // input
	status = clSetKernelArg(combine_kernel, 2, sizeof(cl_int), &num_elements);
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// 실행을 위한 워크 아이템의 인덱스 공간(글로벌 워크 사이즈) 정의
	size_t globalWorkSize[1];
	globalWorkSize[0] = num_elements;
	//================================================================================================================

	if (time) {
		// 커널 실행, event를 이용한 커널 실행 시간 측정
		cl_event event;
		status = clEnqueueNDRangeKernel(ocl.commandQueue, combine_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event);
		if (status != 0) checkError(status, __LINE__);
		status = clWaitForEvents(1, &event);
		if (status != 0) checkError(status, __LINE__);
		status = clFinish(ocl.commandQueue);
		if (status != 0) checkError(status, __LINE__);
		//clReleaseMemObject(inputBuffer);
		cl_ulong time_start;
		cl_ulong time_end;
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		printf("[combine]   dur_time(openCL)           = %6.5f [msec] \n", (time_end - time_start) / 1000000.0);
		//================================================================================================================
		clReleaseKernel(combine_kernel);
	}
	else {
		// 커널 실행
		status = clEnqueueNDRangeKernel(ocl.commandQueue, combine_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
		if (status != 0) checkError(status, __LINE__);
		status = clFinish(ocl.commandQueue);
		if (status != 0) checkError(status, __LINE__);
		//clReleaseMemObject(inputBuffer);
		//================================================================================================================
		clReleaseKernel(combine_kernel);
	}
}


void convert2_opencl(cl_mem outputBuffer, cl_mem inputBuffer, Configs &ccfg, ocl_args_d_t &ocl, cl_int status, bool time = false)
{
	//printf("[convert2]  input [%4d,%4d,%4d,%4d] -> output [%4d,%4d,%4d,%4d]  \n", ccfg.N, ccfg.C, ccfg.H, ccfg.W, ccfg.N, ccfg.K, ccfg.P, ccfg.Q);
	//status = clFinish(ocl.commandQueue);
	//if (status != 0) checkError(status, __LINE__);
	// 커널 생성 (커널 함수 이름을 인자로 전달)
	cl_kernel convert2_kernel = clCreateKernel(ocl.program, "convert2_kernel", &status);
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// 커널에서 사용할 인자 값 계산 작업
	int num_elements = ccfg.N * ccfg.P * ccfg.Q;
	//================================================================================================================

	// 커널 함수 인자 값 전달 (버퍼와 커널 연결)
	status = clSetKernelArg(convert2_kernel, 0, sizeof(cl_mem), &outputBuffer); // output
	status = clSetKernelArg(convert2_kernel, 1, sizeof(cl_mem), &inputBuffer); // input
	status = clSetKernelArg(convert2_kernel, 2, sizeof(cl_int), &ccfg.N);
	status = clSetKernelArg(convert2_kernel, 3, sizeof(cl_int), &ccfg.C);
	status = clSetKernelArg(convert2_kernel, 4, sizeof(cl_int), &ccfg.H);
	status = clSetKernelArg(convert2_kernel, 5, sizeof(cl_int), &ccfg.W);
	status = clSetKernelArg(convert2_kernel, 6, sizeof(cl_int), &num_elements);
	if (status != 0) checkError(status, __LINE__);
	//================================================================================================================

	// 실행을 위한 워크 아이템의 인덱스 공간(글로벌 워크 사이즈) 정의
	size_t globalWorkSize[1];
	globalWorkSize[0] = num_elements;
	//================================================================================================================

	if (time) {
		// 커널 실행, event를 이용한 커널 실행 시간 측정
		cl_event event;
		status = clEnqueueNDRangeKernel(ocl.commandQueue, convert2_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event);
		if (status != 0) checkError(status, __LINE__);
		status = clWaitForEvents(1, &event);
		if (status != 0) checkError(status, __LINE__);
		status = clFinish(ocl.commandQueue);
		if (status != 0) checkError(status, __LINE__);
		//clReleaseMemObject(inputBuffer);
		cl_ulong time_start;
		cl_ulong time_end;
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		printf("[convert2]  dur_time(openCL)           = %6.5f [msec] \n", (time_end - time_start) / 1000000.0);
		//================================================================================================================
		clReleaseKernel(convert2_kernel);
	}
	else {
		// 커널 실행
		status = clEnqueueNDRangeKernel(ocl.commandQueue, convert2_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
		if (status != 0) checkError(status, __LINE__);
		status = clFinish(ocl.commandQueue);
		if (status != 0) checkError(status, __LINE__);
		//clReleaseMemObject(inputBuffer);
		//================================================================================================================
		clReleaseKernel(convert2_kernel);
	}
}