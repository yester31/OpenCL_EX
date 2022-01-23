// This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// OpenCL includes
#include <CL/cl.h>

// kernel을 읽어서 char pointer생성
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

void matMul(float *A, float *B, float *C, int M, int N, int K)
{
	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	for (int m = 0; m < M; ++m) {
		for (int n = 0; n < N; ++n) {
			float sum = 0;
			for (int k = 0; k < K; ++k) {
				sum += A[m * K + k] * B[k * N + n];
			}
			C[m * N + n] = sum;
		}
	}
	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	printf("dur_time(cpu)              = %6.5f [msec] \n", (start_time2 - start_time1) / 1000.f);

}

void valid_results_f(float *result_1, float *result_2, int M, int N) {
	bool result = true;
	for (int i = 0; i < M * N; i++) {
		if ((result_1[i]) != result_2[i]) {
			printf("[%d] The results is not matched! (%f, %f)\n", i, result_1[i], result_2[i]);
			result = false;
		}
	}
	if (result)printf("Results is same!! works well! \n");
	else printf("results is not matched! \n");
}

void print_results(float *output, int M, int N) {
	printf("\n \n");
	for (int m = 0; m < M; ++m) {
		for (int n = 0; n < N; ++n) {
			printf("%.2f ", output[m * N + n]);
		}printf("\n");
	}printf("\n \n");
}

void generate_data_f(float* ptr, unsigned int size) {
	float tt = 1;
	while (size--) {
		*ptr++ = rand() % 10;
		//*ptr++ = tt;
	}
}

// platform 2. Device 1: NVIDIA GeForce RTX 3060 Laptop GPU
// dur_time(openCL) = 5.58694[msec]
// dur_time(cpu) = 1918.76697[msec]

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
	generate_data_f(A, M * K);
	generate_data_f(B, K * N);

	//print_results(A, M, K);
	//print_results(B, K, N);

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
	printf("\n\nSELECT PLATFORM('1' ~ '%d') : ", numPlatforms);
	scanf("%d", &platformNum_);
	printf("\n");
	printf("SELECT DEVICE('1' ~ '%d') : ", numDevices);
	scanf("%d", &deviceNum_);
	printf("\n");

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

	char* programSource = readSource("MatMul.cl"); // 커널 함수 파일 로드
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);	// 프로그램 생성
	status = clBuildProgram(program, numDevices, &devices[deviceNum_ - 1], NULL, NULL, NULL);					// 디바이스를 위한 프로그램을 빌드(컴파일)

	//-----------------------------------------------------
	// STEP 8: Create the kernel
	//----------------------------------------------------- 

	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "matMul_kernel", &status); // 커널 생성 (커널 함수 이름을 인자로 전달)

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

	size_t globalWorkSize[1]; // 실행을 위한 워크 아이템의 인덱스 공간(글로벌 워크 사이즈) 정의
	globalWorkSize[0] = M * N;

	//-----------------------------------------------------
	// STEP 11: Enqueue the kernel for execution
	//----------------------------------------------------- 

	cl_event event;

	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &event); // 커널 실행
	
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

	matMul(A, B, H, M, N, K);
	//print_results(H, M, N);

	valid_results_f(H, C, M, N); // Verify the output
	//print_results(C, M, N);

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