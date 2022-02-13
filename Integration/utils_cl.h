#pragma once
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

// 데이터 랜덤 값 초기화 
template <typename T>
void initDataRandom255(T* ptr, unsigned int size) {
	srand(777); // rand seed 값 전달
	while (size--) {
		*ptr++ = rand() % 255; // (0 - 255 데이터 초기화)
	}
}

// 데이터 랜덤 값 초기화
template <typename T>
void initDataRandomZP1(T* ptr, unsigned int size) {
	srand(777); // rand seed 값 전달
	while (size--) {
		*ptr++ = (rand() % 255) / 255.f; // (0 - 1 데이터 초기화)
	}
}

// 전달 받은 데이터 배열을 파일로 생성
template <typename T>
void tofile(T* Buffer, int data_count, std::string fname = "../Validation_py/Tensor_from_C") {
	std::ofstream fs(fname, std::ios::binary);
	if (fs.is_open())
		fs.write((const char*)Buffer, data_count * sizeof(T));
	fs.close();
	std::cout << "Done! file production to " << fname << std::endl;
}

// 전달 받은 경로의 파일을 로드하여 배열로 생성
template <typename T>
void fromfile(T* Buffer, int data_count, std::string fname = "../Validation_py/Tensor_from_py") {
	std::ifstream ifs(fname, std::ios::binary);
	if (ifs.is_open())
		ifs.read((char*)Buffer, data_count * sizeof(T));
	ifs.close();
	std::cout << "Done! file load from " << fname << std::endl;
}

// 각 레이어에 대한 입력과 출력을 비롯한 여러 파라미터 설정 값
class Configs {

public:
	int N, C, H, W;		// data [N,C,H,W]
	int K, P, Q;		// output [N,K,P,Q]
	int KH, KW;			// weight height, width
	int SH, SW;			// stride 
	int PL, PR, PT, PB; // pad left, right, top, bottom
	int MODE;			// 0 : Conv2d, 1 : Bicubic, 2 : Convert
	int scale_factor;	// for interpolation

	Configs(int N_, int C_, int H_, int W_, int K_, int KH_, int KW_, int SH_ = 1, int SW_ = 1, int PL_ = 0, int PR_ = 0, int PT_ = 0, int PB_ = 0, int MODE_ = 0)
		: N(N_), C(C_), H(H_), W(W_), K(K_), KH(KH_), KW(KW_), SH(SH_), SW(SW_), PL(PL_), PR(PR_), PT(PT_), PB(PB_), MODE(MODE_)
	{
		P = ((H_ + PT_ + PB_ - KH_) / SH_) + 1;
		Q = ((W_ + PL_ + PR_ - KW_) / SW_) + 1;

		printf("cfg input[%4d,%4d,%4d,%4d]  output[%4d,%4d,%4d,%4d]  kernel[%4d,%4d,%4d,%4d]\n", N, C, H, W, N, K, P, Q, K, C, KH, KW);
	};

	Configs(Configs &cfg, int K_, int KH_, int KW_, int SH_ = 1, int SW_ = 1, int PL_ = 0, int PR_ = 0, int PT_ = 0, int PB_ = 0, int MODE_ = 0)
		: K(K_), KH(KH_), KW(KW_), SH(SH_), SW(SW_), PL(PL_), PR(PR_), PT(PT_), PB(PB_)
	{
		N = cfg.N;
		C = cfg.K;
		H = cfg.P;
		W = cfg.Q;
		P = ((H + PT_ + PB_ - KH_) / SH_) + 1;
		Q = ((W + PL_ + PR_ - KW_) / SW_) + 1;

		printf("cfg input[%4d,%4d,%4d,%4d]  output[%4d,%4d,%4d,%4d]  kernel[%4d,%4d,%4d,%4d]\n", N, C, H, W, N, K, P, Q, K, C, KH, KW);
	};

	Configs(int N_, int C_, int H_, int W_, int scale_factor_, int MODE_ = 1)
		: N(N_), C(C_), H(H_), W(W_), K(C_), scale_factor(scale_factor_), MODE(MODE_)
	{
		P = H_ * scale_factor_;
		Q = W_ * scale_factor_;

		printf("cfg input[%4d,%4d,%4d,%4d]  output[%4d,%4d,%4d,%4d]  sclae_factor_(%4d,%4d)\n", N, C, H, W, N, K, P, Q, scale_factor_, scale_factor_);
	};

	Configs(Configs &cfg, int MODE_ = 2)
		: N(cfg.N), C(cfg.K), H(cfg.P), W(cfg.Q), K(cfg.K), P(cfg.P), Q(cfg.Q), MODE(MODE_)
	{
		printf("cfg input[%4d,%4d,%4d,%4d]  output[%4d,%4d,%4d,%4d]\n", N, C, H, W, N, K, P, Q);
	};

	Configs(int N_, int C_, int H_, int W_, int K_, int P_, int Q_, int MODE_ = 4)
		: N(N_), C(C_), H(H_), W(W_), K(K_), P(P_), Q(Q_), MODE(MODE_)
	{
		printf("cfg input[%4d,%4d,%4d,%4d]  output[%4d,%4d,%4d,%4d]\n", N, C, H, W, N, K, P, Q);
	};

	Configs(Configs &cfg, int ic, int oc, int MODE_ = 4)
		: N(cfg.N), C(ic), H(cfg.H), W(cfg.W), K(oc), P(cfg.H), Q(cfg.W), MODE(MODE_)
	{
		printf("cfg input[%4d,%4d,%4d,%4d]  output[%4d,%4d,%4d,%4d]\n", N, C, H, W, N, K, P, Q);
	};

};


// kernel을 읽어서 char pointer생성
char* readSource(char* kernelPath, size_t& size) {

	cl_int status;
	FILE *fp;
	char *source;
	printf("[FILE LOAD] OpenCL file is: %s\n\n", kernelPath);

	fp = fopen(kernelPath, "r");
	if (!fp) {
		printf("[ERROR] Could not open kernel file\n");
		exit(-1);
	}
	status = fseek(fp, 0, SEEK_END);
	if (status != 0) {
		printf("[ERROR] Error seeking to end of file\n");
		exit(-1);
	}
	size = ftell(fp);
	if (size < 0) {
		printf("[ERROR] Error getting file position\n");
		exit(-1);
	}

	rewind(fp);
	source = (char *)malloc(size + 1);

	for (int i = 0; i < size + 1; i++) {
		source[i] = '\0';
	}

	if (source == NULL) {
		printf("[ERROR] Error allocating space for the kernel source\n");
		exit(-1);
	}

	fread(source, 1, size, fp);
	source[size] = '\0';
	fclose(fp);
	return source;
}


// Print an error message to screen (only if it occurs)
void checkError(cl_int error, int line) {
	if (error != CL_SUCCESS) {
		switch (error) {
		case CL_DEVICE_NOT_FOUND:                 printf("-- Error at %d:  Device not found.\n", line); break;
		case CL_DEVICE_NOT_AVAILABLE:             printf("-- Error at %d:  Device not available\n", line); break;
		case CL_COMPILER_NOT_AVAILABLE:           printf("-- Error at %d:  Compiler not available\n", line); break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:    printf("-- Error at %d:  Memory object allocation failure\n", line); break;
		case CL_OUT_OF_RESOURCES:                 printf("-- Error at %d:  Out of resources\n", line); break;
		case CL_OUT_OF_HOST_MEMORY:               printf("-- Error at %d:  Out of host memory\n", line); break;
		case CL_PROFILING_INFO_NOT_AVAILABLE:     printf("-- Error at %d:  Profiling information not available\n", line); break;
		case CL_MEM_COPY_OVERLAP:                 printf("-- Error at %d:  Memory copy overlap\n", line); break;
		case CL_IMAGE_FORMAT_MISMATCH:            printf("-- Error at %d:  Image format mismatch\n", line); break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:       printf("-- Error at %d:  Image format not supported\n", line); break;
		case CL_BUILD_PROGRAM_FAILURE:            printf("-- Error at %d:  Program build failure\n", line); break;
		case CL_MAP_FAILURE:                      printf("-- Error at %d:  Map failure\n", line); break;
		case CL_INVALID_VALUE:                    printf("-- Error at %d:  Invalid value\n", line); break;
		case CL_INVALID_DEVICE_TYPE:              printf("-- Error at %d:  Invalid device type\n", line); break;
		case CL_INVALID_PLATFORM:                 printf("-- Error at %d:  Invalid platform\n", line); break;
		case CL_INVALID_DEVICE:                   printf("-- Error at %d:  Invalid device\n", line); break;
		case CL_INVALID_CONTEXT:                  printf("-- Error at %d:  Invalid context\n", line); break;
		case CL_INVALID_QUEUE_PROPERTIES:         printf("-- Error at %d:  Invalid queue properties\n", line); break;
		case CL_INVALID_COMMAND_QUEUE:            printf("-- Error at %d:  Invalid command queue\n", line); break;
		case CL_INVALID_HOST_PTR:                 printf("-- Error at %d:  Invalid host pointer\n", line); break;
		case CL_INVALID_MEM_OBJECT:               printf("-- Error at %d:  Invalid memory object\n", line); break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  printf("-- Error at %d:  Invalid image format descriptor\n", line); break;
		case CL_INVALID_IMAGE_SIZE:               printf("-- Error at %d:  Invalid image size\n", line); break;
		case CL_INVALID_SAMPLER:                  printf("-- Error at %d:  Invalid sampler\n", line); break;
		case CL_INVALID_BINARY:                   printf("-- Error at %d:  Invalid binary\n", line); break;
		case CL_INVALID_BUILD_OPTIONS:            printf("-- Error at %d:  Invalid build options\n", line); break;
		case CL_INVALID_PROGRAM:                  printf("-- Error at %d:  Invalid program\n", line); break;
		case CL_INVALID_PROGRAM_EXECUTABLE:       printf("-- Error at %d:  Invalid program executable\n", line); break;
		case CL_INVALID_KERNEL_NAME:              printf("-- Error at %d:  Invalid kernel name\n", line); break;
		case CL_INVALID_KERNEL_DEFINITION:        printf("-- Error at %d:  Invalid kernel definition\n", line); break;
		case CL_INVALID_KERNEL:                   printf("-- Error at %d:  Invalid kernel\n", line); break;
		case CL_INVALID_ARG_INDEX:                printf("-- Error at %d:  Invalid argument index\n", line); break;
		case CL_INVALID_ARG_VALUE:                printf("-- Error at %d:  Invalid argument value\n", line); break;
		case CL_INVALID_ARG_SIZE:                 printf("-- Error at %d:  Invalid argument size\n", line); break;
		case CL_INVALID_KERNEL_ARGS:              printf("-- Error at %d:  Invalid kernel arguments\n", line); break;
		case CL_INVALID_WORK_DIMENSION:           printf("-- Error at %d:  Invalid work dimensionsension\n", line); break;
		case CL_INVALID_WORK_GROUP_SIZE:          printf("-- Error at %d:  Invalid work group size\n", line); break;
		case CL_INVALID_WORK_ITEM_SIZE:           printf("-- Error at %d:  Invalid work item size\n", line); break;
		case CL_INVALID_GLOBAL_OFFSET:            printf("-- Error at %d:  Invalid global offset\n", line); break;
		case CL_INVALID_EVENT_WAIT_LIST:          printf("-- Error at %d:  Invalid event wait list\n", line); break;
		case CL_INVALID_EVENT:                    printf("-- Error at %d:  Invalid event\n", line); break;
		case CL_INVALID_OPERATION:                printf("-- Error at %d:  Invalid operation\n", line); break;
		case CL_INVALID_GL_OBJECT:                printf("-- Error at %d:  Invalid OpenGL object\n", line); break;
		case CL_INVALID_BUFFER_SIZE:              printf("-- Error at %d:  Invalid buffer size\n", line); break;
		case CL_INVALID_MIP_LEVEL:                printf("-- Error at %d:  Invalid mip-map level\n", line); break;
		case -1024:                               printf("-- Error at %d:  *clBLAS* Functionality is not implemented\n", line); break;
		case -1023:                               printf("-- Error at %d:  *clBLAS* Library is not initialized yet\n", line); break;
		case -1022:                               printf("-- Error at %d:  *clBLAS* Matrix A is not a valid memory object\n", line); break;
		case -1021:                               printf("-- Error at %d:  *clBLAS* Matrix B is not a valid memory object\n", line); break;
		case -1020:                               printf("-- Error at %d:  *clBLAS* Matrix C is not a valid memory object\n", line); break;
		case -1019:                               printf("-- Error at %d:  *clBLAS* Vector X is not a valid memory object\n", line); break;
		case -1018:                               printf("-- Error at %d:  *clBLAS* Vector Y is not a valid memory object\n", line); break;
		case -1017:                               printf("-- Error at %d:  *clBLAS* An input dimension (M,N,K) is invalid\n", line); break;
		case -1016:                               printf("-- Error at %d:  *clBLAS* Leading dimension A must not be less than the size of the first dimension\n", line); break;
		case -1015:                               printf("-- Error at %d:  *clBLAS* Leading dimension B must not be less than the size of the second dimension\n", line); break;
		case -1014:                               printf("-- Error at %d:  *clBLAS* Leading dimension C must not be less than the size of the third dimension\n", line); break;
		case -1013:                               printf("-- Error at %d:  *clBLAS* The increment for a vector X must not be 0\n", line); break;
		case -1012:                               printf("-- Error at %d:  *clBLAS* The increment for a vector Y must not be 0\n", line); break;
		case -1011:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix A is too small\n", line); break;
		case -1010:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix B is too small\n", line); break;
		case -1009:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix C is too small\n", line); break;
		case -1008:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector X is too small\n", line); break;
		case -1007:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector Y is too small\n", line); break;
		case -1001:                               printf("-- Error at %d:  Code -1001: no GPU available?\n", line); break;
		default:                                  printf("-- Error at %d:  Unknown with code %d\n", line, error);
		}
		exit(1);
	}
}

// OpenCL 세팅 클래스
class ocl_args_d_t
{
public :
	ocl_args_d_t();
	~ocl_args_d_t();

	// Regular OpenCL objects:
	cl_context       context;           // hold the context handler
	cl_device_id*    devices;           // device handlers array ptr
	cl_uint          numDevices;        // searched device handlers
	cl_command_queue commandQueue;      // hold the commands-queue handler
	cl_program       program;           // hold the program handler
	cl_platform_id*  platforms;         // platform handlers array ptr
	cl_uint          numPlatforms;      // 
	cl_event *       events;
	int events_count;
	int platformNum_;					// choiced platform , start from 1
	int deviceNum_;						// choiced device, start from 1
};

ocl_args_d_t::ocl_args_d_t() :context(NULL), devices(NULL), commandQueue(NULL), program(NULL), platforms(NULL),
events(NULL), events_count(0), numDevices(0), platformNum_(2), deviceNum_(1) {}

ocl_args_d_t::~ocl_args_d_t()
{
	cl_int status = CL_SUCCESS;

	if (program) status = clReleaseProgram(program);
	if (status != 0) checkError(status, __LINE__);

	if (commandQueue) status = clReleaseCommandQueue(commandQueue);
	if (status != 0) checkError(status, __LINE__);

	for (int i = 0; i < numDevices; i++) {
		if (devices[i]) status = clReleaseDevice(devices[i]);
		if (status != 0) checkError(status, __LINE__);
	}
	free(devices);

	if (context) status = clReleaseContext(context);
	if (status != 0) checkError(status, __LINE__);

	free(platforms);
}


char* loadClFiles(char**cl_files, int file_count, size_t &fileSizes) {
	printf("\nOpenCL kernel files load!!! \n");
	if (file_count == 1) {
		printf("[%d] OpenCL kernel file : %s\n", 1, cl_files[0]);
		char* source = readSource(cl_files[0], fileSizes); // cl file load
		//printf("cl file print : %s\n", source);
		return source;
	}
	else {
		size_t totalFileSizes = 0;
		for (int i = 0; i < file_count; i++) { // for calculation total cl files size
			printf("[%d] OpenCL kernel file : %s\n", i + 1, cl_files[i]);
			char* sourcest = readSource(cl_files[i], fileSizes); // cl file load for file size
			totalFileSizes += fileSizes;
			free(sourcest);
		}
		char *sourceTotal = (char*)malloc(totalFileSizes);
		for (int i = 0; i < file_count; i++) {
			char* source = readSource(cl_files[i], fileSizes); // cl file load
			if (i == 0) strcpy(sourceTotal, source); // copy first cl file to source_sum
			else strcat(sourceTotal, source); // source_sum 뒤에 sources[i]을 붙임
			free(source);
		}
		fileSizes = totalFileSizes;
		//printf("cl file print : %s\n", sourceTotal);
		return sourceTotal;
	}
}

void cl_setting(ocl_args_d_t &ocl, cl_int &status, char *clKernelSource, size_t &clKernelSource_size, bool cl_online_compile = true, bool manual_selection = false) {
	printf("[SETTING] OpenCL settring START !!!\n");
	status = clGetPlatformIDs(0, NULL, &ocl.numPlatforms);								// 사용 가능한 플랫폼 수 확인
	ocl.platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * ocl.numPlatforms);	// 플랫폼 정보를 가져올 수 있도록 공간 할당
	status = clGetPlatformIDs(ocl.numPlatforms, ocl.platforms, NULL);					// 플랫폼 정보를 가져옴
	if (status != 0) checkError(status, __LINE__);

	char* value;					// devcie 정보 문자열
	size_t valueSize;				// devcie 정보 문자열 크기
	cl_uint maxComputeUnits;

	if (manual_selection) {
		for (int i = 0; i < ocl.numPlatforms; i++) {
			// get all devices
			status = clGetDeviceIDs(ocl.platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &ocl.numDevices);
			ocl.devices = (cl_device_id*)malloc(sizeof(cl_device_id) * ocl.numDevices);
			status = clGetDeviceIDs(ocl.platforms[i], CL_DEVICE_TYPE_ALL, ocl.numDevices, ocl.devices, NULL);
			// for each device print critical attributes
			for (int j = 0; j < ocl.numDevices; j++) {
				// print device name
				status = clGetDeviceInfo(ocl.devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
				value = (char*)malloc(valueSize);
				status = clGetDeviceInfo(ocl.devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
				printf("platform %d. Device %d: %s\n", i + 1, j + 1, value);
				free(value);
				// print hardware device version
				status = clGetDeviceInfo(ocl.devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
				value = (char*)malloc(valueSize);
				status = clGetDeviceInfo(ocl.devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
				printf(" %d.%d Hardware version: %s\n", i + 1, 1, value);
				free(value);
				// print software driver version
				status = clGetDeviceInfo(ocl.devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
				value = (char*)malloc(valueSize);
				status = clGetDeviceInfo(ocl.devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
				printf(" %d.%d Software version: %s\n", i + 1, 2, value);
				free(value);
				// print c version supported by compiler for device
				status = clGetDeviceInfo(ocl.devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
				value = (char*)malloc(valueSize);
				status = clGetDeviceInfo(ocl.devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
				printf(" %d.%d OpenCL C version: %s\n", i + 1, 3, value);
				free(value);
				// print parallel compute units
				status = clGetDeviceInfo(ocl.devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
				printf(" %d.%d Parallel compute units: %d\n", i + 1, 4, maxComputeUnits);
			}
		}
		printf("\n\nSELECT PLATFORM('1' ~ '%d') : ", ocl.numPlatforms);
		scanf("%d", &ocl.platformNum_);
		printf("\n");
		printf("SELECT DEVICE('1' ~ '%d') : ", ocl.numDevices);
		scanf("%d", &ocl.deviceNum_);
		printf("\n");
	}
	else {
		status = clGetDeviceIDs(ocl.platforms[ocl.platformNum_ - 1], CL_DEVICE_TYPE_ALL, 0, NULL, &ocl.numDevices);
		ocl.devices = (cl_device_id*)malloc(sizeof(cl_device_id) * ocl.numDevices);
		status = clGetDeviceIDs(ocl.platforms[ocl.platformNum_ - 1], CL_DEVICE_TYPE_ALL, ocl.numDevices, ocl.devices, NULL);
		if (status != 0) checkError(status, __LINE__);

		status = clGetDeviceInfo(ocl.devices[ocl.deviceNum_ - 1], CL_DEVICE_NAME, 0, NULL, &valueSize);
		value = (char*)malloc(valueSize);
		status = clGetDeviceInfo(ocl.devices[ocl.deviceNum_ - 1], CL_DEVICE_NAME, valueSize, value, NULL);
		if (status != 0) checkError(status, __LINE__);
		printf("platform %d. Device %d: %s\n", ocl.platformNum_, ocl.deviceNum_, value);
		free(value);
	}

	// 선택한 디바이스 정보를 가져옴
	status = clGetDeviceIDs(ocl.platforms[ocl.platformNum_ - 1], CL_DEVICE_TYPE_ALL, ocl.numDevices, ocl.devices, NULL);
	if (status != 0) checkError(status, __LINE__);

	// context 생성 및 (원하는)디바이스와 연결
	ocl.context = clCreateContext(NULL, 1, &ocl.devices[ocl.deviceNum_ - 1], NULL, NULL, &status);
	if (status != 0) checkError(status, __LINE__);

	// 명령어 큐 생성 및 (원하는)디바이스와 연결
	ocl.commandQueue = clCreateCommandQueue(ocl.context, ocl.devices[ocl.deviceNum_ - 1], CL_QUEUE_PROFILING_ENABLE, &status);	
	if (status != 0) checkError(status, __LINE__);

	if (cl_online_compile) {
		ocl.program = clCreateProgramWithSource(ocl.context, 1, (const char**)&clKernelSource, NULL, &status);	// 프로그램 생성
		status = clBuildProgram(ocl.program, ocl.numDevices, &ocl.devices[ocl.deviceNum_ - 1], NULL, NULL, NULL);	// 디바이스를 위한 프로그램을 빌드(컴파일)
	}
	else {
		cl_int binary_status;
		ocl.program = clCreateProgramWithBinary(ocl.context, 1, &ocl.devices[ocl.deviceNum_ - 1], (const size_t *)&clKernelSource_size, (const unsigned char**)&clKernelSource, &binary_status, &status);	// 프로그램 생성
	}
	printf("[SETTING] OpenCL settring END !!!\n\n");
	if (status != 0) checkError(status, __LINE__);
}