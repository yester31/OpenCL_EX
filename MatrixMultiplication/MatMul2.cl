// TODO: Add OpenCL kernel code here.

#define TILE_WIDTH 32

__kernel 
void matMul_kernel_sm(
	__global float *A,
	__global float *B,
	__global float *C,
	int M, int N, int K
)
{
	__local float s_A[TILE_WIDTH * TILE_WIDTH];
	__local float s_B[TILE_WIDTH * TILE_WIDTH];

	//int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int idx = get_group_id(0) ;
	if (idx >= M * N) return;

	int col = idx % N;
	int row = idx / N;
	int tx = idx % TILE_WIDTH;
	int ty = col / TILE_WIDTH;
	float sum = 0.f;

	for (int t = 0; t < (K - 1) / TILE_WIDTH + 1; t++)
	{
		if (row < M && t*TILE_WIDTH + tx < N)
			s_A[ty * TILE_WIDTH + tx] = A[row * K + t * TILE_WIDTH + tx];
		else
			s_A[ty * TILE_WIDTH + tx] = 0.f;

		if (t*TILE_WIDTH + ty < N && col < N)
			s_B[ty * TILE_WIDTH + tx] = B[(t * TILE_WIDTH + ty) * K + col];
		else
			s_B[ty * TILE_WIDTH + tx] = 0.f;

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int i = 0; i < TILE_WIDTH; i++)
			sum += s_A[ty * TILE_WIDTH + i] * s_B[i * TILE_WIDTH + tx];
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (row < M && col < N)
		C[col + row * N] = sum;
}