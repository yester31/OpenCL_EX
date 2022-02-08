#pragma once
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <io.h>

// convolution 용 configuration struct
typedef struct {
	int N, C, H, W;	// data [N,C,H,W]
	int K, P, Q;	// output [N,K,P,Q]
	int KH, KW;		// weight height, width
	int SH, SW;		// stride 
	int PL, PR, PT, PB; // pad left, right, top, bottom
} Conv2dConfig;

template <typename T>
void tofile(T* Buffer, int data_count, std::string fname = "../Validation_py/Tensor_from_C") {
	std::ofstream fs(fname, std::ios::binary);
	if (fs.is_open())
		fs.write((const char*)Buffer, data_count * sizeof(T));
	fs.close();
	std::cout << "Done! file production to " << fname << std::endl;
}

template <typename T>
void fromfile(T* Buffer, int data_count, std::string fname = "../Validation_py/Tensor_from_py") {
	std::ifstream ifs(fname, std::ios::binary);
	if (ifs.is_open())
		ifs.read((char*)Buffer, data_count * sizeof(T));
	ifs.close();
	std::cout << "Done! file load from " << fname << std::endl;
}

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

// 결과값 비교
void compareResults(float *result_1, float *result_2, int size) {
	bool result = true;
	for (int i = 0; i < size; i++) {
		if ((result_1[i]) != result_2[i]) {
			printf("[%d] The results is not matched! (%f, %f)\n", i, result_1[i], result_2[i]);
			result = false;
		}
	}
	if (result)printf("Results is same!! works well! \n");
	else printf("results is not matched! \n");
}

// Tensor 출력
void printData(float* valueCheckInput, int input_n = 1, int input_c = 1, int input_h = 1, int input_w = 1, int offset = 0) {
	if (offset == 1) { input_n = 1; }

	int temp1 = input_w * input_h * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				printf("  ");
				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int g_idx = w_idx + temp4;

					printf("%6.2f ", valueCheckInput[g_idx]);

				}printf("\n");
			}printf("\n\n");
		}printf("\n");
	}printf("\n");
}

// 데이터 초기화(스칼라 값) Default = 1
template <typename T>
void initDataScalar(T* ptr, unsigned int size, float tt = 1) {
	while (size--) {
		*ptr++ = tt;
	}
}

// 데이터 초기화(1부터 1씩 증가)
void initDataStep(float* ptr, unsigned int size) {
	float tt = 1;
	while (size--) {
		*ptr++ = tt++;
	}
}

// 데이터 랜덤 값 초기화 
// rand() 0 ~ 32767 사이 난수 생성
void initDataRandom(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = rand() % 10; // (0 - 10 사이 float 데이터 초기화)
	}
}

// 데이터 랜덤 값 초기화 
template <typename T>
void initDataRandom255(T* ptr, unsigned int size) {
	char tt = 1;
	srand(777); // rand seed 값 전달
	while (size--) {
		*ptr++ = rand() % 255; // (0 - 255 사이 char 데이터 초기화)
	}
}


// CPU에서 행렬곱 연산 (matrix multiplication on cpu)
void matMulCpu(float *A, float *B, float *C, int M, int N, int K)
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

// CPU에서 Convoltuion 연산
void convolution(float* output, float* input, float* weight, int IN, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW) {
	printf("===== Conventional Convolution ===== \n");
	uint64_t start_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	int OH = ((IH - KH) / SH) + 1;
	int OW = ((IW - KW) / SW) + 1;

	int C_offset_i, C_offset_o, C_offset_k, H_offset_i, H_offset_o, H_offset_k, W_offset_i, W_offset_o, W_offset_k, ⁠g_idx_i, g_idx_o, g_idx_k;
	int N_offset_i = IC * IH * IW;
	int N_offset_o = OC * OH * OW;
	int N_offset_k = IC * KH * KW;

	for (int ⁠n_idx = 0; ⁠n_idx < IN; ⁠n_idx++) {
		C_offset_i = ⁠n_idx * N_offset_i;
		C_offset_o = ⁠n_idx * N_offset_o;

		for (int k_idx = 0; k_idx < OC; k_idx++) {
			C_offset_k = k_idx * N_offset_k;
			H_offset_o = k_idx * OH * OW + C_offset_o;

			for (int ⁠c_idx = 0; ⁠c_idx < IC; ⁠c_idx++) {
				H_offset_i = ⁠c_idx * IH * IW + C_offset_i;
				H_offset_k = ⁠c_idx * KH * KW + C_offset_k;

				for (int rowStride = 0; rowStride < OH; rowStride++) {
					W_offset_o = rowStride * OW + H_offset_o;

					for (int colStride = 0; colStride < OW; colStride++) {
						float sum = 0;
						g_idx_o = colStride + W_offset_o;

						for (int y = rowStride * SH; y < rowStride * SH + KH; y++) {
							W_offset_i = y * IW + H_offset_i;
							W_offset_k = (y - rowStride * SH) * KH + H_offset_k;

							for (int x = colStride * SW; x < colStride * SW + KW; x++) {

								⁠g_idx_i = x + W_offset_i;
								g_idx_k = (x - colStride * SW) + W_offset_k;
								sum += input[⁠g_idx_i] * weight[g_idx_k];

							}
						}

						output[g_idx_o] += sum;

					}
				}
			}
		}
	}
	uint64_t start_time2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	printf("dur_time(cpu)              = %6.5f [msec] \n", (start_time2 - start_time1) / 1000.f);

}

// CPU에서 Tensor zero padding 연산
void zeroPadding(float*  zeroPaddingOutput, float*  zeroPaddingInput, int input_n, int input_c, int input_h, int input_w, int leftPadingSize, int rightPadingSize, int topPadingSize, int bottomPadingSize) {

	int temp1 = input_w * input_h * input_c;
	int temp1o = (input_h + topPadingSize + bottomPadingSize) * (input_w + leftPadingSize + rightPadingSize) * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		int temp2o = ⁠n_idx * temp1o;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			int temp3o = ⁠c_idx * (input_w + leftPadingSize + rightPadingSize) * (input_h + topPadingSize + bottomPadingSize) + temp2o;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				int temp4o = (⁠h_idx + topPadingSize) * (input_w + leftPadingSize + rightPadingSize) + leftPadingSize + temp3o;

				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int ⁠g_idx = w_idx + temp4;
					int g_idx_Output = w_idx + temp4o;
					zeroPaddingOutput[g_idx_Output] = zeroPaddingInput[⁠g_idx];
				}
			}
		}
	}
}

//파일 이름 가져오기(DFS) window용
int SearchFile(const std::string& folder_path, std::vector<std::string> &file_names, bool recursive = false) {
	_finddata_t file_info;
	std::string any_file_pattern = folder_path + "\\*";
	intptr_t handle = _findfirst(any_file_pattern.c_str(), &file_info);
	if (handle == -1) {
		std::cerr << "folder path not exist: " << folder_path << std::endl;
		return -1;
	}
	do {
		std::string file_name = file_info.name;
		if (recursive) {
			if (file_info.attrib & _A_SUBDIR) {//check whtether it is a sub direcotry or a file
				if (file_name != "." && file_name != "..") {
					std::string sub_folder_path = folder_path + "//" + file_name;
					SearchFile(sub_folder_path, file_names);
					std::cout << "a sub_folder path: " << sub_folder_path << std::endl;
				}
			}
			else {
				std::string file_path = folder_path + "/" + file_name;
				file_names.push_back(file_path);
			}
		}
		else {
			if (!(file_info.attrib & _A_SUBDIR)) {//check whtether it is a sub direcotry or a file
				std::string file_path = folder_path + "/" + file_name;
				file_names.push_back(file_path);
			}
		}
	} while (_findnext(handle, &file_info) == 0);
	_findclose(handle);
	return 0;
}

// 전처리 : BGR -> RGB, NHWC->NCHW, Normalize (0 ~ 1)
void preprocess(std::vector<float> &output, std::vector<uint8_t> &input, int IN, int IC, int IH, int IW) {

	int C_offset, H_offset, W_offset, g_in, g_out;
	int N_offset = IH * IW * IC;
	for (int ⁠n_idx = 0; ⁠n_idx < IN; ⁠n_idx++) {
		H_offset = ⁠n_idx * N_offset;
		for (int ⁠h_idx = 0; ⁠h_idx < IH; ⁠h_idx++) {
			W_offset = ⁠h_idx * IW * IC + H_offset;
			for (int w_idx = 0; w_idx < IW; w_idx++) {
				C_offset = w_idx * IC + W_offset;
				for (int ⁠c_idx = 0; ⁠c_idx < IC; ⁠c_idx++) {
					g_in = C_offset + 2 - ⁠c_idx;
					g_out = H_offset + ⁠c_idx * IH * IW + ⁠h_idx * IW + w_idx;
					output[g_out] = static_cast <float>(input[g_in]) / 255.f;
				}
			}
		}
	}
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