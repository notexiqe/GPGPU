#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define N (1024)
#define SH_DIM 32

#define CUDA_CHECK_RETURN(value){\
		cudaError_t _m_cudaStat = value;\
		if (_m_cudaStat != cudaSuccess) {\
			fprintf(stderr, "Error %s at line %d in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
			exit(1);\
		} }

typedef unsigned int uint;

void print_matrix(int* a, int n);

__global__ void transport_matrix(int* a, int* b) {

    int idX = threadIdx.x + blockDim.x * blockIdx.x,
        idY = threadIdx.y + blockDim.y * blockIdx.y,
        size = blockDim.x * gridDim.x;

    b[idX * size + idY] = a[idX + idY * size];
}

__host__ float kernel_transport_matrix(dim3 gridDim, dim3 blockDim, int n) {

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *ha, *hb,
        *da, *db;

    ha = (int*)malloc(n * n * sizeof(int));
    hb = (int*)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            ha[i * n + j] = (int)(rand() % 100);
        }
    }

    CUDA_CHECK_RETURN(cudaMalloc(&da, n * n * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&db, n * n * sizeof(int)));

    CUDA_CHECK_RETURN(cudaMemcpy(da, ha, n * n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(db, hb, n * n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEventRecord(start, 0);

    transport_matrix<<<gridDim,blockDim>>>(da, db);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaEventElapsedTime(&elapsedTime, start, stop);

    CUDA_CHECK_RETURN(cudaMemcpy(hb, db, n * n * sizeof(int), cudaMemcpyDeviceToHost));
    
    return elapsedTime;
}

__global__ void transport_matrix_sh_mem(int* a, int* b, int n) {

    __shared__ int buffer[SH_DIM][SH_DIM];

    int idX = threadIdx.x + blockDim.x * blockIdx.x,
        idY = threadIdx.y + blockDim.y * blockIdx.y,
        size = blockDim.x * gridDim.x;

    buffer[threadIdx.y][threadIdx.x] = a[idX + idY * size];
    __syncthreads();

    idX = threadIdx.x + blockIdx.y * blockDim.x;
    idY = threadIdx.y + blockIdx.x * blockDim.y;

    b[idX + idY * size] = buffer[threadIdx.x][threadIdx.y];
}

__host__ float kernel_transport_matrix_sh_mem(dim3 gridDim, dim3 blockDim, int n) {

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *ha, *hb,
        *da, *db;

    ha = (int*)malloc(n * n * sizeof(int));
    hb = (int*)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            hb[i * n + j] = ha[i * n + j] = (int)(rand() % 100);
        }
    }

    CUDA_CHECK_RETURN(cudaMalloc(&da, n * n * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&db, n * n * sizeof(int)));

    CUDA_CHECK_RETURN(cudaMemcpy(da, ha, n * n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEventRecord(start, 0);

    transport_matrix_sh_mem<<<gridDim,blockDim>>>(da, db, n);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaEventElapsedTime(&elapsedTime, start, stop);

    CUDA_CHECK_RETURN(cudaMemcpy(hb, db, n * n * sizeof(int), cudaMemcpyDeviceToHost));

    return elapsedTime;
}

__global__ void transport_matrix_off_bank_conflict(int* a, int* b, int n) {

    __shared__ int buffer[SH_DIM][SH_DIM + 1];

    int idX = threadIdx.x + blockDim.x * blockIdx.x,
        idY = threadIdx.y + blockDim.y * blockIdx.y,
        size = blockDim.x * gridDim.x;

    buffer[threadIdx.y][threadIdx.x] = a[idX + idY * size];
    __syncthreads();

    idX = threadIdx.x + blockIdx.y * blockDim.x;
    idY = threadIdx.y + blockIdx.x * blockDim.y;

    b[idX + idY * size] = buffer[threadIdx.x][threadIdx.y];
}

__host__ float kernel_transport_matrix_off_bank_conflict(dim3 gridDim, dim3 blockDim, int n) {

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *ha, *hb,
        *da, *db;

    ha = (int*)malloc(n * n * sizeof(int));
    hb = (int*)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            hb[i * n + j] = ha[i * n + j] = (int)(rand() % 100);
        }
    }

    CUDA_CHECK_RETURN(cudaMalloc(&da, n * n * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&db, n * n * sizeof(int)));

    cudaEventRecord(start, 0);

    transport_matrix_off_bank_conflict<<<gridDim,blockDim>>>(da, db, n);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaEventElapsedTime(&elapsedTime, start, stop);

    CUDA_CHECK_RETURN(cudaMemcpy(hb, db, n * n * sizeof(int), cudaMemcpyDeviceToHost));

    return elapsedTime;
}

int main() {

    int threadsPerBlock = 32,
        blocksPerGridDimX = (int)(ceilf(N / (float)threadsPerBlock)),
        blocksPerGridDimY = (int)(ceilf(N / (float)threadsPerBlock));

    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);
    dim3 gridDim(blocksPerGridDimX, blocksPerGridDimY, 1);

    printf("Default transportation matrix\t\t- %.4f ms;\n", kernel_transport_matrix(gridDim, blockDim, N));
    printf("Transportation matrix w/ Sh mem\t\t- %.4f ms;\n", kernel_transport_matrix_sh_mem(gridDim, blockDim, N));
    printf("Transportation matrix w/o Bank Conflict\t- %.4f ms;\n", kernel_transport_matrix_off_bank_conflict(gridDim, blockDim, N));

    printf("\n");
    return 0;
}