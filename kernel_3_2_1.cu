#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

#define N 4096

#define CUDA_CHECK_RETURN(value){\
		cudaError_t _m_cudaStat = value;\
		if (_m_cudaStat != cudaSuccess) {\
			fprintf(stderr, "Error %s at line %d in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
			exit(1);\
		} }

__global__ void gTest1(float* a) {
    int idX = threadIdx.x + blockIdx.x * blockDim.x;
    int idY = threadIdx.y + blockIdx.y * blockDim.y;
    int I = gridDim.x * blockDim.x;

    a[idX + idY * I] = (float)(threadIdx.x + blockDim.y * blockIdx.x);
}

__global__ void gTest2(float* a) {
    int idX = threadIdx.x + blockIdx.x * blockDim.x;
    int idY = threadIdx.y + blockIdx.y * blockDim.y;
    int J = gridDim.y * blockDim.y;

    a[idY + idX * J] = (float)(threadIdx.y + threadIdx.x * blockDim.y);
}

float init_gTest1() {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float elapsedTime;

    int threadsPerBlock = 32,
        blocksPerGridDimX = N / threadsPerBlock,
        blocksPerGridDimY = N / threadsPerBlock;

    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1),
         gridDim(blocksPerGridDimX, blocksPerGridDimY, 1);

    float *a;

    CUDA_CHECK_RETURN(cudaMalloc(&a, N * N * sizeof(float)));

    cudaEventRecord(start, 0);

    gTest1<<<gridDim,
             blockDim>>>(a);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(a);

    return elapsedTime;
}

__host__ float init_gTest2() {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsedTime;

    int threadsPerBlock = 32,
        blocksPerGridDimX = N / threadsPerBlock,
        blocksPerGridDimY = N / threadsPerBlock;

    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1),
         gridDim(blocksPerGridDimX, blocksPerGridDimY, 1);

    float* b;

    CUDA_CHECK_RETURN(cudaMalloc(&b, N * N * sizeof(float)));

    cudaEventRecord(start, 0);

    gTest2<<<gridDim,
             blockDim>>>(b);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(b);

    return elapsedTime;
}


int main() {

    printf("gTest1 - %lf ms\n", init_gTest1());
    printf("gTest2 - %lf ms\n", init_gTest2());

    printf("\n");
    return 0;
}