// VS_Project
// CUDA Headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include <stdlib.h>
#include <stdio.h>

#define V_SIZE_MIN (1<<10)
#define V_SIZE_MAX (1<<23)

#define CUDA_CHECK_RETURN(value){										\
		cudaError_t _m_cudaStat = value;								\
		if (_m_cudaStat != cudaSuccess) {								\
			fprintf(stderr, "Error %s at line %d in file %s\n",			\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
			exit(1);													\
		} }

// kernel-function
__global__ void d_vector_add(int* a, int* b, int* c, int n) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x; // thread index definition

	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}

__host__ LONGLONG d_fill_vector_cuda_event(int blocksPerGrid, int threadsPerBlock, int n) {

	cudaEvent_t start, stop; // CUDA timer

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int* ha, * hb, * hc, // for memory allocate on HOST
		* da, * db, * dc; // for memory allocate on DEVICE

	// allocate HOST memory space
	ha = (int*)malloc(n * sizeof(int));
	hb = (int*)malloc(n * sizeof(int));
	hc = (int*)malloc(n * sizeof(int));

	// allocate DEVICE memory space
	CUDA_CHECK_RETURN(cudaMallocManaged(&da, n * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged(&db, n * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged(&dc, n * sizeof(int)));

	for (int i = 0; i < n; i++) {
		ha[i] = i;
		hb[i] = i;
		hc[i] = 0;
	}

	// copy data from HOST to DEVICE memory
	CUDA_CHECK_RETURN(cudaMemcpy(da, ha, n * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(db, hb, n * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(dc, hc, n * sizeof(int), cudaMemcpyHostToDevice));

	cudaEventRecord(start, 0);

	d_vector_add<<<blocksPerGrid,
					threadsPerBlock>>>(da, db, dc, n);

	cudaEventRecord(stop, 0);

	cudaMemcpy(hc, dc, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	LONGLONG microseconds = LONGLONG(milliseconds * 1000);

	//
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	// free DEVICE memory space
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	// free HOST memory space
	free(ha);
	free(hb);
	free(hc);

	return microseconds;
}

__host__ LONGLONG d_fill_vector_query_counter(int blocksPerGrid, int threadsPerBlock, int n) {

	LARGE_INTEGER start, end, elapsed;
	LARGE_INTEGER frequency;

	int* ha, * hb, * hc, // for memory allocate on HOST
		* da, * db, * dc; // for memory allocate on DEVICE

	// allocate HOST memory space
	ha = (int*)malloc(n * sizeof(int));
	hb = (int*)malloc(n * sizeof(int));
	hc = (int*)malloc(n * sizeof(int));

	// allocate DEVICE memory space
	CUDA_CHECK_RETURN(cudaMallocManaged(&da, n * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged(&db, n * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged(&dc, n * sizeof(int)));

	for (int i = 0; i < n; i++) {
		ha[i] = i;
		hb[i] = i;
		hc[i] = 0;
	}

	// copy data from HOST to DEVICE memory
	CUDA_CHECK_RETURN(cudaMemcpy(da, ha, n * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(db, hb, n * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(dc, hc, n * sizeof(int), cudaMemcpyHostToDevice));

	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);

	d_vector_add<<<blocksPerGrid,
					threadsPerBlock>>>(da, db, dc, n);


	//cudaMemcpy(hc, dc, n * sizeof(int), cudaMemcpyDeviceToHost);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

	QueryPerformanceCounter(&end);

	elapsed.QuadPart = end.QuadPart - start.QuadPart;
	elapsed.QuadPart *= 1000000; // microseconds
	elapsed.QuadPart /= frequency.QuadPart;



	// free DEVICE memory space
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	// free HOST memory space
	free(ha);
	free(hb);
	free(hc);

	return elapsed.QuadPart;
}

__host__ void d_case() {

	printf("\nCOMPUTING ON DEVICE\n");

	LONGLONG cuda_t, win_t;

	LONGLONG N = 1 << 10;
	int iPower = 10;

	for (int i = 1 << 6; i <= 1 << 9; i <<= 1) {
		printf("\n\tblock size - %d;\n", i);
		while (N <= 1 << 23) {
			cuda_t = d_fill_vector_cuda_event(N / i, i, N);
			win_t = d_fill_vector_query_counter(N / i, i, N);
			printf("size - (2<<%d); CUDA Event - %5lld us\t(%lld ms); WinAPI timer - %5lld us\t(%lld ms)\n", iPower++, cuda_t, (cuda_t / 1000), win_t, (win_t / 1000));
			N <<= 1;
		}
		iPower = 10;
		N = 1 << 10;
	}
}

int main() {

	d_case();

	return 0;
}