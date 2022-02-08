
// Convert2 (NCHW->NHWC, YCbCr->BGR)
__kernel void ycbcr2bgr_kernel(
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
