#pragma once

void bicubic_opencl(cl_mem outputBuffer, cl_mem inputBuffer, Configs &ccfg, ocl_args_d_t &ocl, cl_int status)
{
	cl_kernel bicubic_kernel = clCreateKernel(ocl.program, "bicubic_kernel", &status); // Ŀ�� ���� (Ŀ�� �Լ� �̸��� ���ڷ� ����)
	if (status != 0) checkError(status, __LINE__);
	float h_scale = float(ccfg.H - 1) / (ccfg.P - 1);
	float w_scale = float(ccfg.W - 1) / (ccfg.Q - 1);
	int num_elements = ccfg.N * ccfg.C * ccfg.P * ccfg.Q;
	// Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(bicubic_kernel, 0, sizeof(cl_mem), &outputBuffer); // output
	status = clSetKernelArg(bicubic_kernel, 1, sizeof(cl_mem), &inputBuffer); // input
	status = clSetKernelArg(bicubic_kernel, 2, sizeof(cl_float), &h_scale); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(bicubic_kernel, 3, sizeof(cl_float), &w_scale); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(bicubic_kernel, 4, sizeof(cl_int), &ccfg.N); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(bicubic_kernel, 5, sizeof(cl_int), &ccfg.C); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(bicubic_kernel, 6, sizeof(cl_int), &ccfg.H); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(bicubic_kernel, 7, sizeof(cl_int), &ccfg.W); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(bicubic_kernel, 8, sizeof(cl_int), &ccfg.P); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(bicubic_kernel, 9, sizeof(cl_int), &ccfg.Q); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(bicubic_kernel, 10, sizeof(cl_int), &num_elements); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	if (status != 0) checkError(status, __LINE__);

	size_t globalWorkSize[1]; // ������ ���� ��ũ �������� �ε��� ����(�۷ι� ��ũ ������) ����
	globalWorkSize[0] = num_elements;

	if (0) {
		status = clEnqueueNDRangeKernel(ocl.commandQueue, bicubic_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL); // Ŀ�� ����
		if (status != 0) checkError(status, __LINE__);
		status = clFinish(ocl.commandQueue);
		if (status != 0) checkError(status, __LINE__);
	}
	else {
		cl_event event;
		status = clEnqueueNDRangeKernel(ocl.commandQueue, bicubic_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event); // Ŀ�� ����
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
	}
}
