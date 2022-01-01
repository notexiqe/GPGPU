// VS_Project
// CUDA Headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include <stdlib.h>
#include <stdio.h>

#define V_SIZE_MIN (1<<10)
#define V_SIZE_MAX (1<<23)

__host__ void print_device_info() {
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	printf("%s\n", deviceProp.name);
	printf("Total amount of constant memory: %zu bytes\n", deviceProp.totalConstMem);
	printf("Total amount of shared memory per block %zu bytes\n", deviceProp.sharedMemPerBlock);
	printf("Total number of registers available per block %lu bytes\n", deviceProp.regsPerBlock);
	printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
	printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("Warp size: %d\n\n", deviceProp.warpSize);

}

// kernel-function
// __global__ [attribute] - called from CPU, execution on GPU 
__global__ void d_vector_add(int* a, int* b, int* c, int n) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x; // thread index definition

	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}

__host__ LONGLONG d_fill_vector(int blocksPerGrid, int threadsPerBlock, int n) {


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int* ha, * hb, * hc, // for memory allocate on HOST
		* da, * db, * dc; // for memory allocate on DEVICE

	// allocate HOST memory space
	ha = (int*)malloc(n * sizeof(int));
	hb = (int*)malloc(n * sizeof(int));
	hc = (int*)malloc(n * sizeof(int));

	// allocate DEVICE memory space
	cudaMallocManaged(&da, n * sizeof(int));
	cudaMallocManaged(&db, n * sizeof(int));
	cudaMallocManaged(&dc, n * sizeof(int));

	for (int i = 0; i < n; i++) {
		ha[i] = i;
		hb[i] = i;
		hc[i] = 0;
	}

	// copy data from HOST to DEVICE memory
	cudaMemcpy(da, ha, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dc, hc, n * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start);

	d_vector_add << <blocksPerGrid,
		threadsPerBlock >> > (da, db, dc, n);

	cudaEventRecord(stop);

	cudaMemcpy(hc, dc, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);

	cudaDeviceSynchronize();

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	//
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	LONGLONG microseconds = LONGLONG(milliseconds * 1000);


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

__host__ void d_case() {

	printf("\nCOMPUTING ON DEVICE\n");

	LONGLONG t;
	LONGLONG N = 1 << 10;
	int iPower = 10;

	for (int i = 1 << 6; i <= 1 << 9; i <<= 1) {
		printf("\n\tblock size - %d;\n", i);
		while (N <= 1 << 23) {
			t = d_fill_vector(N / i, i, N);
			printf("size - (2<<%d); time - %5lld us\t(%lld ms)\n", iPower++, t, (t / 1000));
			N <<= 1;
		}
		iPower = 10;
		N = 1 << 10;
	}
}


__host__ LONGLONG h_add_vector(int n) {

	LARGE_INTEGER start, end, elapsed;
	LARGE_INTEGER frequency;

	int* a, * b, * c;

	a = (int*)malloc(n * sizeof(int));
	b = (int*)malloc(n * sizeof(int));
	c = (int*)malloc(n * sizeof(int));

	for (int i = 0; i < n; i++) {
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);

	for (int i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}

	QueryPerformanceCounter(&end);

	elapsed.QuadPart = end.QuadPart - start.QuadPart;
	elapsed.QuadPart *= 1000000; // microseconds
	elapsed.QuadPart /= frequency.QuadPart;

	free(a);
	free(b);
	free(c);

	return elapsed.QuadPart;
}

__host__ void h_case() {

	printf("\nCOMPUTING ON HOST\n\n");

	LONGLONG t;

	int iPower = 10U;

	for (int n = V_SIZE_MIN; n <= V_SIZE_MAX; n = n << 1, iPower++) {

		printf("size - (1<<%d); ", iPower);
		t = h_add_vector(n);
		printf("time - %5lld us\t(%2lld ms)\n", t, (t / 1000));
	}
}

int main() {

	print_device_info();

	d_case();

	h_case();

	return 0;
}