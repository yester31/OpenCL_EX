// TODO: Add OpenCL kernel code here.
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
