#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)

__global__ void kernel(int* a, int* b, int* c) {

    int i = threadIdx.x;

    c[i] = a[i] + b[i];
}

__global__ void kernel2(int* a, int* b, int* c) {
    __shared__ float cache[256];
    int i, tid = threadIdx.x + blockIdx.x * blockDim.x, cacheIndex = threadIdx.x, temp = 0;

    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
    __syncthreads();

    i = blockDim.x / 2;

    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}


int main() {
    srand(time(NULL));

    cudaDeviceProp prop;
    int i, res, whichDevice;
    int* dev_a, * dev_a_p, * h_a, * h_b, * h_a_p, * h_b_p, * h_c_p;
    int* dev_a0, * dev_b0, * dev_c0, * dev_a1, * dev_b1, * dev_c1;
    float elapsed_time;
    cudaStream_t stream0, stream1;
    cudaEvent_t start, stop;

    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        printf("Device does not support overlapping\n");
        return 0;
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_a = (int*)malloc(FULL_DATA_SIZE * sizeof(int));
    h_b = (int*)malloc(FULL_DATA_SIZE * sizeof(int));
    for (int i = 0; i < FULL_DATA_SIZE; i++)
        h_a[i] = rand() % 10000;
    cudaMalloc((void**)&dev_a, FULL_DATA_SIZE * sizeof(int));
    cudaEventRecord(start, 0);
    cudaMemcpy(dev_a, h_a, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time host->device: %f\n", elapsed_time);

    cudaEventRecord(start, 0);
    cudaMemcpy(h_b, dev_a, FULL_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time device->host: %f\n", elapsed_time);

    cudaHostAlloc((void**)&h_a_p, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_b_p, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_c_p, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void**)&dev_a_p, FULL_DATA_SIZE * sizeof(int));
    for (i = 0; i < FULL_DATA_SIZE; i++)
        h_a_p[i] = rand() % 10000;
    cudaEventRecord(start, 0);
    cudaMemcpy(dev_a_p, h_a_p, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time paged-locked host->device: %f\n", elapsed_time);

    cudaEventRecord(start, 0);
    cudaMemcpy(h_b_p, dev_a_p, FULL_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time paged-locked device->host: %f\n", elapsed_time);
    cudaFree(dev_a);
    cudaFree(dev_a_p);


    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaMalloc((void**)&dev_a0, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b0, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c0, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_a1, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b1, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c1, FULL_DATA_SIZE * sizeof(int));
    for (i = 0; i < FULL_DATA_SIZE; i++) {
        h_a_p[i] = i;
        h_b_p[i] = i;
    }
    cudaEventRecord(start, 0);
    for (i = 0; i < FULL_DATA_SIZE; i += N * 2) {
        cudaMemcpyAsync(dev_a0, h_a_p + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_a1, h_a_p + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_b0, h_b_p + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_b1, h_b_p + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        kernel << < 1, N, 0, stream0 >> > (dev_a0, dev_b0, dev_c0);
        kernel << < 1, N, 0, stream1 >> > (dev_a1, dev_b1, dev_c1);
        cudaMemcpyAsync(h_c_p + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(h_c_p + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    }
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time addition vectors: %f\n", elapsed_time);

    cudaEventRecord(start, 0);
    for (i = 0; i < FULL_DATA_SIZE; i += N * 2) {
        cudaMemcpyAsync(dev_a0, h_a_p + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_a1, h_a_p + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_b0, h_b_p + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_b1, h_b_p + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        kernel2 << < 16, N, 0, stream0 >> > (dev_a0, dev_b0, dev_c0);
        kernel2 << < 16, N, 0, stream1 >> > (dev_a1, dev_b1, dev_c1);
        cudaMemcpyAsync(h_c_p + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(h_c_p + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    }
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    for (i = 0, res = 0; i < FULL_DATA_SIZE; i++)
        res += h_c_p[i];
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time multiplication vectors: %f\n", elapsed_time);

    cudaFree(dev_a0);
    cudaFree(dev_a1);
    cudaFree(dev_b0);
    cudaFree(dev_b1);
    cudaFree(dev_c0);
    cudaFree(dev_c1);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaFree(h_a_p);
    cudaFree(h_b_p);
    cudaFree(h_c_p);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n");
    return 0;
}