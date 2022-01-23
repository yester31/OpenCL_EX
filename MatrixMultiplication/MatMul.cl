
// OpenCL kernel to perform an element-wise 
// Matrix multiplication                       

__kernel
void matMul_kernel(
	__global float *A,
	__global float *B,
	__global float *C,
	int M, int N, int K
	)
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
