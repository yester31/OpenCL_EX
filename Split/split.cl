// TODO: Add OpenCL kernel code here.
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


// split (NCHW->N1HW)
//__kernel void split_first_channel_kernel(
//	__global float *output1,
//	__global float *output2,
//	__global float *input,
//	int N, int C, int H, int W,
//	__global float *input,
//	int num_elements)
//{
//	// Get the work-item¡¯s unique ID
//	int index = get_global_id(0);
//	if (index >= num_elements) return;
//
//	output1[index] = input[index];
//	output2[index] = input[index + H * W];
//	output2[index + H * W] = input[index + 2 * H * W];
//}
