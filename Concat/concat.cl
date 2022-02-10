// Concatenates 
__kernel void concat_kernel(
	__global float *output,
	__global float *input1,
	__global float *input2,
	int N, int C, int H, int W, int K, int axis,
	int num_elements)
{
	int tid = get_global_id(0);
	if (tid >= num_elements) return;

	//int w_idx = tid % W;
	//tid /= W;
	//int h_idx = tid % H;
	//tid /= H;
	//int c_idx = tid % C;
	//int b_idx = tid / C;

	//int in_idx = b_idx * C * H * W + c_idx * H * W + h_idx * W + w_idx;

	//output[tid] = input1[in_idx];
	//output[tid + C * H * W] = input2[in_idx];

	output[tid] = input1[tid];
	output[tid + C * H * W] = input2[tid];
}