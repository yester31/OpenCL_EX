// Upsample

// Nearnest Neighbor Interpolation
__kernel void nearnest_neighbor_kernel(
	__global float *output,
	__global float *input,
	int h_scale,
	int w_scale,
	int N, int C, int H, int W, int P, int Q,
	int num_elements)
{
	int tid0 = get_global_id(0);
	if (tid0 >= num_elements) return;
	int tid = tid0;

	int q_idx = tid % Q;
	tid /= Q;
	int p_idx = tid % P;
	tid /= P;
	int c_idx = tid % C;
	int b_idx = tid / C;

	int h_idx = p_idx / h_scale;
	int w_idx = q_idx / w_scale;

	int in_idx = b_idx * C * H * W + c_idx * H * W + h_idx * W + w_idx;

	output[tid0] = input[in_idx];
}