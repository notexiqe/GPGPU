#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 8192

#define CUDA_CHECK_RETURN(value){\
		cudaError_t _m_cudaStat = value;\
		if (_m_cudaStat != cudaSuccess) {\
			fprintf(stderr, "Error %s at line %d in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
			exit(1);\
		} }

void print_m(int* a, int n) {

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            printf("%2d ", a[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int check_tran_m(int* a, int* b, int n) {

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (a[i * n + j] - b[j * n + i] != 0){
                printf("\n\tError: Fail transpose of matrix;\n");
                return 1;
            }
        }
    }
    return 0;
}

__global__ void tran_m(int* a, int* b, int n) {

    int idX, idY;

    idX = threadIdx.x + blockDim.x * blockIdx.x;
    idY = threadIdx.y + blockDim.y * blockIdx.y;

    b[idY * n + idX] = a[idX * n + idY];
}

int kernel(dim3 gridDim, dim3 blockDim, float* returnTime, int n) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    int *ha, *hb, 
        *da, *db;

    float elapsedTime;

    ha = (int*)malloc(n * n * sizeof(int)); // for create
    hb = (int*)malloc(n * n * sizeof(int)); // for copy results from device

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            hb[i * n + j] = ha[i * n + j] = (int)(rand() % 100);
        }
    }

    //printf("Original matrix\n");
    //print_m(ha, n);

    CUDA_CHECK_RETURN(cudaMalloc(&da, n * n * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&db, n * n * sizeof(int)));

    CUDA_CHECK_RETURN(cudaMemcpy(da, ha, n * n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEventRecord(start, 0);

    tran_m<<<gridDim,
            blockDim>>>(da, db, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaEventElapsedTime(&elapsedTime, start, stop);

    CUDA_CHECK_RETURN(cudaMemcpy(hb, db, n * n * sizeof(int), cudaMemcpyDeviceToHost));

    if (check_tran_m(ha, hb, n))
        return 1;
    *returnTime = elapsedTime;

    printf("Matrix transposition time: %lf\n", elapsedTime);
    //print_m(hb, n);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(da);
    cudaFree(db);

    free(ha);
    free(hb);

    return 0;
}

int main() {

    srand((unsigned int)time(NULL));

    int threadsPerBlock = 32,
        blocksPerGridDimX = (int)(ceilf(N / (float)threadsPerBlock)),
        blocksPerGridDimY = (int)(ceilf(N / (float)threadsPerBlock));;

    float time = 0;

    dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);
    dim3 gridDim(blocksPerGridDimX, blocksPerGridDimY, 1);

    printf("\tN = %d\n\n", N);
    if (kernel(gridDim, blockDim, &time, N))
        return 1;

    printf("\n");
    return 0;
}