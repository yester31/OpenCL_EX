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
__kernel void bicubic2d_kernel(
	__global float *output,
	__global float *input,
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
