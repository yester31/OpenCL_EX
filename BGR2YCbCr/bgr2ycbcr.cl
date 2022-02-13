
// Convert1 (NHWC->NCHW, BGR->YCbCr)
__kernel void bgr2ycbcr_kernel(
	__global float *output,
	__global unsigned char *input,
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

	int iidx  = n_idx * H * W * C + h_idx * W * C + w_idx * C;
	int gidx0 = n_idx * C * H * W +             h_idx * W + w_idx;
	int gidx1 = n_idx * C * H * W +     H * W + h_idx * W + w_idx;
	int gidx2 = n_idx * C * H * W + 2 * H * W + h_idx * W + w_idx;
	output[gidx0] = 16.f  + ((64.738   * input[iidx + 2]) + (129.057 * input[iidx + 1]) + (25.064   * input[iidx])) / 256.f;
	output[gidx1] = 128.f + ((-37.945f * input[iidx + 2]) - (74.494f * input[iidx + 1]) + (112.439f * input[iidx])) / 256.f;
	output[gidx2] = 128.f + ((112.439f * input[iidx + 2]) - (94.154f * input[iidx + 1]) - (18.285f  * input[iidx])) / 256.f;
}

// Convert1 (NHWC, BGR->YCbCr)
__kernel void bgr2ycbcr_kernel2(
	__global float *output,
	__global unsigned char *input,
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
	output[iidx] = 16.f + ((64.738   * input[iidx + 2]) + (129.057 * input[iidx + 1]) + (25.064   * input[iidx])) / 256.f;
	output[iidx+1] = 128.f + ((-37.945f * input[iidx + 2]) - (74.494f * input[iidx + 1]) + (112.439f * input[iidx])) / 256.f;
	output[iidx+2] = 128.f + ((112.439f * input[iidx + 2]) - (94.154f * input[iidx + 1]) - (18.285f  * input[iidx])) / 256.f;
}
