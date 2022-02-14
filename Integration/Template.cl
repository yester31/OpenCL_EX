// bicubic interpolation kernel

__inline float cubic1d(float x0, float x1, float x2, float x3, float t) {
	float A = -0.75f;
	float coeffs_0 = ((A * (t + 1.f) - 5.f * A) * (t + 1.f) + 8.f * A) * (t + 1.f) - 4.f * A;
	float coeffs_1 = ((A + 2.f) * t - (A + 3.f)) * t * t + 1.f;
	float coeffs_2 = ((A + 2.f) * (1.f - t) - (A + 3.f)) * (1.f - t) * (1.f - t) + 1.f;
	float coeffs_3 = ((A * (2.f - t) - 5.f * A) * (2.f - t) + 8.f * A) * (2.f - t) - 4.f * A;
	return x0 * coeffs_0 + x1 * coeffs_1 + x2 * coeffs_2 + x3 * coeffs_3;
}

// 1. bicubic2d
__kernel void bicubic_kernel(
	__global float *output,
	//__global float *input,
	__global unsigned char *input,
	float h_scale,
	float w_scale,
	int N, int C, int H, int W, int P, int Q,
	int num_elements)
{
	// Get the work-item¡¯s unique ID
	int index = get_global_id(0);
	if (index >= num_elements) return;

	int q_idx = index % Q; // Q 
	int idx = index / Q;
	int p_idx = idx % P; // P 
	idx /= P;
	int c_idx = idx % C; // C
	int n_idx = idx / C; // N
	const int g_idx = n_idx * C * P * Q + c_idx * P * Q + p_idx * Q + q_idx;

	// just copy
	if (H == P && W == Q) {
		output[g_idx] = input[g_idx];
		return;
	}

	// Interpolation kernel
	float real_x = w_scale * q_idx;
	int in_x = floor(real_x);
	float t_x = real_x - in_x;

	float real_y = h_scale * p_idx;
	int in_y = floor(real_y);
	float t_y = real_y - in_y;

	float coeff[4];
	int access_x0 = max(min((in_x - 1), W - 1), 0);
	int access_x1 = max(min((in_x + 0), W - 1), 0);
	int access_x2 = max(min((in_x + 1), W - 1), 0);
	int access_x3 = max(min((in_x + 2), W - 1), 0);
	int cu_idx = n_idx * C * H * W + c_idx * H * W;
#pragma unroll
	for (int k = 0; k < 4; k++) {
		int access_y = max(min((in_y - 1 + k), H - 1), 0);
		coeff[k] = cubic1d(
			input[cu_idx + access_y * W + access_x0],
			input[cu_idx + access_y * W + access_x1],
			input[cu_idx + access_y * W + access_x2],
			input[cu_idx + access_y * W + access_x3],
			t_x);
	}

	output[g_idx] = cubic1d(coeff[0], coeff[1], coeff[2], coeff[3], t_y);
}


// Convert1 (NHWC->NCHW, BGR->YCbCr), bgr2ycbcr_kernel
__kernel void convert1_kernel(
	__global float *output,
	__global float *input,
	int N, int C, int H, int W,
	int num_elements)
{
	// Get the work-item¡¯s unique ID
	int index = get_global_id(0);
	if (index >= num_elements) return;

	int w_idx = index % W; // W
	int idx = index / W;
	int h_idx = idx % H; // H
	int n_idx = idx / H; // N

	int iidx = n_idx * H * W * C + h_idx * W * C + w_idx * C;
	int gidx0 = n_idx * C * H * W + h_idx * W + w_idx;
	int gidx1 = n_idx * C * H * W + H * W + h_idx * W + w_idx;
	int gidx2 = n_idx * C * H * W + 2 * H * W + h_idx * W + w_idx;
	output[gidx0] = 16.f + ((64.738   * input[iidx + 2]) + (129.057 * input[iidx + 1]) + (25.064   * input[iidx])) / 256.f;
	output[gidx1] = 128.f + ((-37.945f * input[iidx + 2]) - (74.494f * input[iidx + 1]) + (112.439f * input[iidx])) / 256.f;
	output[gidx2] = 128.f + ((112.439f * input[iidx + 2]) - (94.154f * input[iidx + 1]) - (18.285f  * input[iidx])) / 256.f;
}

// split (NCHW->N1HW)
__kernel void split_first_channel_kernel(
	__global float *output,
	__global float *input,
	int num_elements)
{
	// Get the work-item¡¯s unique ID
	int index = get_global_id(0);
	if (index >= num_elements) return;

	output[index] = input[index];
}

// split (N3HW -> N1HW, N2HW)
__kernel void split_kernel(
	__global float *output1,
	__global float *output2,
	__global float *input,
	int N, int C, int H, int W,
	int num_elements)
{
	// Get the work-item¡¯s unique ID
	int index = get_global_id(0);
	if (index >= num_elements) return;

	output1[index] = input[index];

	output2[index] = input[index + H * W];
	output2[index + H * W] = input[index + 2 * H * W];
}

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
	// Get the work-item¡¯s unique ID
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
	// Get the work-item¡¯s unique ID
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
	__global float *bias,
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

	output[s_idx] = input[n_idx] + bias[k_idx];
}

// split (N1HW->N3HW)
__kernel void combine_first_channel_kernel(
	__global float *output,
	__global float *input,
	int num_elements)
{
	// Get the work-item¡¯s unique ID
	int index = get_global_id(0);
	if (index >= num_elements) return;

	output[index] = input[index];
}


// Convert2 (NCHW->NHWC, YCbCr->BGR), ycbcr2bgr_kernel
__kernel void convert2_kernel(
	__global float *output,
	__global float *input,
	int N, int C, int H, int W,
	int num_elements)
{
	// Get the work-item¡¯s unique ID
	int index = get_global_id(0);
	if (index >= num_elements) return;

	int w_idx = index % W; // W
	int idx = index / W;
	int h_idx = idx % H; // H
	int n_idx = idx / H; // N

	int iidx = n_idx * H * W * C + h_idx * W * C + w_idx * C;

	int gidx0 = n_idx * C * H * W + h_idx * W + w_idx;
	int gidx1 = n_idx * C * H * W + H * W + h_idx * W + w_idx;
	int gidx2 = n_idx * C * H * W + 2 * H * W + h_idx * W + w_idx;

	output[iidx] = 298.082f * input[gidx2] / 256.f + 516.412f * input[gidx1] / 256.f - 276.836f;
	output[iidx + 1] = 298.082f * input[gidx2] / 256.f - 100.291f * input[gidx1] / 256.f - 208.120f * input[gidx0] / 256. + 135.576f;
	output[iidx + 2] = 298.082f * input[gidx2] / 256.f + 408.583f * input[gidx0] / 256.f - 222.921f;
}

// 2. Naive Convolution 
__kernel void conv2d_kernel(
	__global float *output,
	__global float *input,
	__global float *weight,
	__global float *bias,
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

	int offset_i0 = n_idx * C * H * W;
	int offset_w0 = k_idx * C * KH * KW;

	for (int c_idx = 0; c_idx < C; c_idx++) {
		int offset_i = c_idx * H * W + offset_i0;
		int offset_w = c_idx * KH * KW + offset_w0;

		for (int y = p_idx * SH; y < p_idx * SH + KH; y++) {
			for (int x = q_idx * SW; x < q_idx * SW + KW; x++) {

				if (y >= top && x >= left && y < (H + top) && x < (W + left)) {

					int i_idx = (x - left) + (y - top) * W + offset_i;
					int w_idx = (x - q_idx * SW) + (y - p_idx * SH) * KH + offset_w;

					output[tid] += input[i_idx] * weight[w_idx];
				}
			}
		}
	}
	output[tid] += bias[k_idx];
}