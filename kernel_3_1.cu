#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cmath>

#define CUDA_CHECK_RETURN(value){\
		cudaError_t _m_cudaStat = value;\
		if (_m_cudaStat != cudaSuccess) {\
			fprintf(stderr, "Error %s at line %d in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
			exit(1);\
		} }

typedef unsigned int uint;

//const uint V_SIZE = 1024*1024*512; // 1 << 29
//const uint V_SIZE = 1024*1024*32; // 1 << 25
const uint V_SIZE = 1024 * 1024; // 1 << 20
const uint MAX_NUM_OF_THREADS = 1024;

__global__ void d_vector_add(int* a, int* b) {

    int idX = threadIdx.x + blockDim.x * blockIdx.x;

    if (idX < V_SIZE) {
        a[idX] = idX;
        b[idX] = V_SIZE - idX;
        a[idX] += b[idX];
    }
}

__host__ float d_test(int threadsPerBlock, int numOfBlocks) {

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *da, *db;
    float elapsedTime;

    CUDA_CHECK_RETURN(cudaMalloc((void**)&da, V_SIZE * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&db, V_SIZE * sizeof(int)));

    cudaEventRecord(start, 0);

    d_vector_add<<<numOfBlocks,
                   threadsPerBlock>>>(da, db);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(da);
    cudaFree(db);

    return elapsedTime;
}


int main() {

    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    float time;

    int numOfBlocks,
        result,        // Occupancy in terms of active blocks
        activeWarps,
        maxWarps;

    for (uint threadsPerBlock = 32; threadsPerBlock <= MAX_NUM_OF_THREADS; threadsPerBlock += 32) {
       numOfBlocks = V_SIZE / threadsPerBlock;
       time = d_test(threadsPerBlock, numOfBlocks);
       printf("threads_per_block = %4d; num_of_blocks = %8d; time = %4.2f ms; ", threadsPerBlock, numOfBlocks, time);

       // Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#occupancy-calculator
       // Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g5a5d67a3c907371559ba692195e8a38c
       // 
       // Theoretical Occupancy
       // The following code sample calculates the occupancy of MyKernel.
       // It then reports the occupancy level with the ratio between concurrent warps versus maximum warps per multiprocessor.
       cudaOccupancyMaxActiveBlocksPerMultiprocessor(
           &result,
           (void*)d_vector_add,
           threadsPerBlock,
           0);
       activeWarps = result * threadsPerBlock / prop.warpSize;
       maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

       printf("\tActive warps - %d; Max warps - %d; ", activeWarps, maxWarps);
       printf("Theoretical Occupancy : %.2lf%\n", (double)activeWarps / maxWarps * 100);

    }

    printf("\n");
    return 0;
}