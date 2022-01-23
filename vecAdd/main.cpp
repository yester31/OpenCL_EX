// This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
#include <CL/cl.h>

// kernel�� �о char pointer����
char* readSource(char* kernelPath) {

	cl_int status;
	FILE *fp;
	char *source;
	long int size;

	printf("Program file is: %s\n", kernelPath);

	fp = fopen(kernelPath, "rb");
	if (!fp) {
		printf("Could not open kernel file\n");
		exit(-1);
	}
	status = fseek(fp, 0, SEEK_END);
	if (status != 0) {
		printf("Error seeking to end of file\n");
		exit(-1);
	}
	size = ftell(fp);
	if (size < 0) {
		printf("Error getting file position\n");
		exit(-1);
	}

	rewind(fp);

	source = (char *)malloc(size + 1);

	int i;
	for (i = 0; i < size + 1; i++) {
		source[i] = '\0';
	}

	if (source == NULL) {
		printf("Error allocating space for the kernel source\n");
		exit(-1);
	}

	fread(source, 1, size, fp);
	source[size] = '\0';

	return source;
}


int main() {
	// This code executes on the OpenCL host

	// Host data
	int *A = NULL;  // Input array
	int *B = NULL;  // Input array
	int *C = NULL;  // Output array

	// Elements in each array
	const int elements = 10;

	// Compute the size of the data 
	size_t datasize = sizeof(int)*elements;

	// Allocate space for input/output data
	A = (int*)malloc(datasize);
	B = (int*)malloc(datasize);
	C = (int*)malloc(datasize);
	// Initialize the input data
	for (int i = 0; i < elements; i++) {
		A[i] = i;
		B[i] = i*2;
	}

	//================================================================================================================
	// Use this to check the output of each API call
	cl_int status;

	//-----------------------------------------------------
	// STEP 1: Discover and initialize the platforms
	//-----------------------------------------------------

	cl_uint numPlatforms = 0;
	cl_platform_id *platforms = NULL;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);							// ��� ������ �÷��� �� Ȯ��
	platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));	// �÷��� ������ ������ �� �ֵ��� ���� �Ҵ�
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);					// �÷��� ������ ������

	//-----------------------------------------------------
	// STEP 2: Discover and initialize the devices
	//----------------------------------------------------- 

	cl_uint numDevices = 0;
	cl_device_id *devices = NULL;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL,	&numDevices);		// �ش� �÷����� ��� ������ ����̽� �� Ȯ��
	devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));						// �� ����̽��� ���� ���� �Ҵ�
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);	// ����̽� ������ ������

	//-----------------------------------------------------
	// STEP 3: Create a context
	//----------------------------------------------------- 

	cl_context context = NULL;
	context = clCreateContext(NULL,	numDevices,	devices, NULL, NULL, &status);	// context ���� �� (���ϴ�)����̽��� ����

	//-----------------------------------------------------
	// STEP 4: Create a command queue
	//----------------------------------------------------- 

	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);	//��ɾ� ť ���� �� (���ϴ�)����̽��� ����

	//================================================================================================================

	//-----------------------------------------------------
	// STEP 5: Create device buffers
	//----------------------------------------------------- 

	cl_mem bufferA;  // Input array on the device
	cl_mem bufferB;  // Input array on the device
	cl_mem bufferC;  // Output array on the device
	bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);	// ����̽� ���� ��ü ����(�Է¿�)
	bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);	// ����̽� ���� ��ü ����(�Է¿�)
	bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);	// ����̽� ���� ��ü ����(��¿�)

	//-----------------------------------------------------
	// STEP 6: Write host data to device buffers
	//----------------------------------------------------- 

	status = clEnqueueWriteBuffer(cmdQueue, bufferA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);	// host (A) -> device (bufferA)����
	status = clEnqueueWriteBuffer(cmdQueue, bufferB, CL_FALSE, 0, datasize, B, 0, NULL, NULL);	// host (B) -> device (bufferB)����

	//-----------------------------------------------------
	// STEP 7: Create and compile the program
	//----------------------------------------------------- 

	char* programSource = readSource("vecAdd.cl");																// Ŀ�� �Լ� ���� �ε�
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);	// ���α׷� ����
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);									// ����̽��� ���� ���α׷��� ����(������)

	//-----------------------------------------------------
	// STEP 8: Create the kernel
	//----------------------------------------------------- 

	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "vecadd", &status); // Ŀ�� ���� (Ŀ�� �Լ� �̸��� ���ڷ� ����)

	//-----------------------------------------------------
	// STEP 9: Set the kernel arguments
	//----------------------------------------------------- 

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC); // Ŀ�� �Լ� ���� �� ���� (���ۿ� Ŀ�� ����)

	//-----------------------------------------------------
	// STEP 10: Configure the work-item structure
	//----------------------------------------------------- 

	size_t globalWorkSize[1]; // ������ ���� ��ũ �������� �ε��� ����(�۷ι� ��ũ ������) ����
	globalWorkSize[0] = elements;

	//-----------------------------------------------------
	// STEP 11: Enqueue the kernel for execution
	//----------------------------------------------------- 

	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL); // Ŀ�� ����

	//-----------------------------------------------------
	// STEP 12: Read the output buffer back to the host
	//----------------------------------------------------- 

	status = clEnqueueReadBuffer(cmdQueue, bufferC, CL_TRUE, 0, datasize, C, 0, NULL, NULL); // device (bufferC) -> host (C) ����

	// Verify the output
	bool result = true;
	for (int i = 0; i < elements; i++) {
		printf("%d + %d = %d\n", A[i], B[i], C[i]);
		if (C[i] !=  A[i] +  B[i]) {
			result = false;
			break;
		}
	}
	if (result) {
		printf("Output is correct\n");
	}
	else {
		printf("Output is incorrect\n");
	}

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
	free(platforms);
	free(devices);
}