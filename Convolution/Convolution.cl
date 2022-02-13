// GEMM Convolution
#define TS 32

// 1. im2col
__kernel void im2col_kernel(
	__global float *output,
	__global float *input,
	int N, 
	int K, int P, int Q, 
	int C, int H, int W, 
	int KH, int KW, 
	int SH, int SW, 
	int left, int top)
{
	// Get the work-item’s unique ID
	int tid = get_global_id(0);
	if (tid >= C * KH * KW * N * P * Q) return;

	int q_idx = tid % Q;
	int idx = tid / Q;
	int p_idx = idx % P;
	idx /= P;
	int b_idx = idx % N;
	idx /= N;
	int kw_idx = idx % KW;
	idx /= KW;
	int kh_idx = idx % KH;
	idx /= KH;

	int k_idx = kw_idx + kh_idx * KW;
	int c_idx = idx % C;
	int w_idx = q_idx * SW - left + kw_idx;
	int h_idx = p_idx * SH - top + kh_idx;

	int n_index2 = c_idx * P * Q * N * KH * KW
		+ k_idx * P * Q * N
		+ b_idx * P * Q
		+ p_idx * Q
		+ q_idx;

	if (w_idx < 0 || w_idx >= W || h_idx < 0 || h_idx >= H) {
		output[tid] = 0.f;
	}
	else {
		int s_idx = b_idx * C * H * W + c_idx * H * W + h_idx * W + w_idx;
		output[n_index2] = input[s_idx];
	}
}


// 2. matrix multiplication
__kernel void matMul_kernel(
	__global float *A,
	__global float *B,
	__global float *C,
	int M, int N, int K)
{
	// Get the work-item’s unique ID
	int idx = get_global_id(0);
	if (idx >= M * N) return;

	int w_idx = idx % N;
	int h_idx = idx / N;
	float sum = 0.f;
	for (int k = 0; k < K; ++k) {
		sum += A[h_idx * K + k] * B[k * N + w_idx];
	}
	C[h_idx * N + w_idx] = sum;
}

// 2. col2im
__kernel void col2im_kernel(
	__global float *output,
	__global float *input,
	int N, int K, int P, int Q)
{
	int tid = get_global_id(0);
	if (tid >= N * K  *P * Q) return;

	int q_idx = tid % Q;// Q 
	int idx = tid / Q;
	int p_idx = idx % P;// P 
	idx /= P;
	int k_idx = idx % K;// K
	int b_idx = idx / K;// N

	int s_idx = b_idx * K * P * Q + k_idx * P * Q + p_idx * Q + q_idx;
	int n_idx = k_idx * N * P * Q + b_idx * P * Q + p_idx * Q + q_idx;

	output[s_idx] = input[n_idx];
}

// on construct....
__kernel void conv2d_kernel(
	__global float *output,
	__global float *input,
	__global float *weight,
	int N, int C, int H, int W,
	int K, int P, int Q,
	int KH, int KW,
	int SH, int SW,
	int left, int top) {

	int tid = get_global_id(0);
	if (tid >= N * K  *P * Q) return;

	int q_idx = tid % Q;// Q 
	int idx = tid / Q;
	int p_idx = idx % P;// P 
	idx /= P;
	int k_idx = idx % K;// K
	int n_idx = idx / K;// N

	float sum = 0;
	for (int c_idx = 0; c_idx < C; c_idx++) {
		int offset_i = c_idx * H * W + n_idx * C * H * W;
		int offset_w = c_idx * KH * KW + k_idx * C * KH * KW;

		for (int y = p_idx * SH; y < p_idx * SH + KH; y++) {
			for (int x = q_idx * SW; x < q_idx * SW + KW; x++) {

				int i_idx = (x + left) + (y + top) * W + offset_i;

				int w_idx = (x - q_idx * SW) + (y - p_idx * SH) * KH + offset_w;

				sum += input[i_idx] * weight[w_idx];
			}
		}
		output[tid] += sum;
	}
}