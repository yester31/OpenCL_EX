#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <io.h>
// 결과값 비교
void compareResults(float *result_1, float *result_2, int size, float loc = 4) {
	bool result = true;
	for (int i = 0; i < size; i++) {
		int tmp = result_1[i];
		int count = 0;
		while (tmp / 10) {
			count++;
			tmp /= 10;
		}
		float eps = pow(0.1, loc - count);
		if (abs(result_1[i] - result_2[i]) > eps) {
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
void initDataScalar(float* ptr, unsigned int size, float tt = 1) {
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

// 데이터 랜덤 값 초기화 
void initDataRandomZP1(float* ptr, unsigned int size) {
	char tt = 1;
	srand(777); // rand seed 값 전달
	while (size--) {
		*ptr++ = (rand() % 255) / 255.f; // (0 - 255 사이 char 데이터 초기화)
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