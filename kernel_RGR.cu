#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cublas_v2.h>
#pragma comment (lib, "cublas.lib")


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#define CUDA_CHECK_RETURN(value){\
		cudaError_t _m_cudaStat = value;\
		if (_m_cudaStat != cudaSuccess) {\
			fprintf(stderr, "Error %s at line %d in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
			exit(1);\
		} }

#define N (1<<23)

__global__ void cuda_saxpy(float alpha, float* x, float* y) {
	int idX = threadIdx.x + blockIdx.x * blockDim.x;
	y[idX] = alpha * x[idX] + y[idX];
}

struct functor {
	const float a;
	functor(float _a) : a(_a) {}
	__host__ __device__ float operator()(float x, float y) {
		return a * x + y;
	}
};

void thrust_saxpy(float a, thrust::device_vector<float>& x, thrust::device_vector<float>& y) {
	functor func(a);
	thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), func);
}


int main() {
	cudaEvent_t start, stop;
	float elapsedTime;

	float* x_d, * x_h,
		 * y_d, * y_h;

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	x_h = (float*)calloc(N, sizeof(float));
	y_h = (float*)calloc(N, sizeof(float));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&x_d, N * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&y_d, N * sizeof(float)));

	for (int i = 0; i < N; i++) {
		x_h[i] = float(i);
		y_h[i] = 0.87f;
	}
	
	CUDA_CHECK_RETURN(cudaMemcpy(x_d, x_h, N * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(y_d, y_h, N * sizeof(float), cudaMemcpyHostToDevice));


	cudaEventRecord(start, 0);

	cuda_saxpy<<< N / 256, 256 >>>(3.0, x_d, y_d);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	CUDA_CHECK_RETURN(cudaMemcpy(y_h, y_d, N * sizeof(float), cudaMemcpyDeviceToHost));
	printf("SAXPY, \"raw\" CUDA, computation time: %.3f ms\n", elapsedTime);

	cudaFree(y_d);
	cudaFree(x_d);
	free(y_h);
	free(x_h);

	// CuBLAS

	float* cx_d, * cx_h,
		 * cy_d, * cy_h;

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	CUDA_CHECK_RETURN(cudaMallocHost((void**)&cx_h, N * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&cy_h, N * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&cx_d, N * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&cy_d, N * sizeof(float)));

	for (int i = 0; i < N; i++) {
		cx_h[i] = float(i);
		cy_h[i] = 0.87f;
	}

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	const int num_rows = N;
	const int num_cols = 1;
	const size_t elem_size = sizeof(float);

	cublasSetMatrix(num_rows, num_cols, elem_size, cx_h, num_rows, cx_d, num_rows);
	cublasSetMatrix(num_rows, num_cols, elem_size, cy_h, num_rows, cy_d, num_rows);

	const int stride = 1;
	float alpha = 3.0f;

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	cublasSaxpy(cublas_handle, N, &alpha, cx_d, stride, cy_d, stride);

	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	cublasGetMatrix(num_rows, num_cols, elem_size, cx_d, num_rows, cx_h, num_rows);
	cublasGetMatrix(num_rows, num_cols, elem_size, cy_d, num_rows, cy_h, num_rows);
	printf("SAXPY, \tcuBLAS\t , computation time: %.3f ms\n", elapsedTime);

	cublasDestroy(cublas_handle);
	CUDA_CHECK_RETURN(cudaFreeHost(cx_h));
	CUDA_CHECK_RETURN(cudaFreeHost(cy_h));
	CUDA_CHECK_RETURN(cudaFree(cx_d));
	CUDA_CHECK_RETURN(cudaFree(cy_d));
	// THRUST

	thrust::host_vector<float> h1(N);
	thrust::host_vector<float> h2(N);
	thrust::sequence(h1.begin(), h1.end());
	thrust::fill(h2.begin(), h2.end(), 0.87);

	thrust::device_vector<float> d1 = h1;
	thrust::device_vector<float> d2 = h2;

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	thrust_saxpy(3.0, d1, d2);
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));

	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));

	h2 = d2;
	h1 = d1;

	printf("SAXPY, \tthrust\t , computation time: %.3f ms\n\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}