#include "optical_flow.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// CUDA Kernel for absolute difference calculation and block reduction
__global__ void motion_magnitude_kernel(const unsigned char* prev, const unsigned char* curr, float* output_sum, int size) {
    extern __shared__ float sdata[];
    
    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float diff = 0.0f;
    if (idx < size) {
        diff = (float)abs((int)curr[idx] - (int)prev[idx]);
    }
    sdata[tx] = diff;
    __syncthreads();
    
    // Perform block-level reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tx < s) {
            sdata[tx] += sdata[tx + s];
        }
        __syncthreads();
    }
    
    // Atomically accumulate block sum to the global sum
    if (tx == 0) {
        atomicAdd(output_sum, sdata[0]);
    }
}

extern "C" {

void* cuda_malloc(size_t size) {
    void* dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return dev_ptr;
}

void cuda_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

int cuda_memcpy_to_device(void* dest, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyHostToDevice failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_memcpy_to_host(void* dest, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyDeviceToHost failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

float calculate_motion_magnitude(const unsigned char* prev_host, const unsigned char* curr_host, int width, int height) {
    int size = width * height;
    if (size <= 0) return 0.0f;
    
    thread_local static unsigned char* d_prev = nullptr;
    thread_local static unsigned char* d_curr = nullptr;
    thread_local static float* d_sum = nullptr;
    thread_local static int cached_size = 0;
    
    cudaError_t err;
    if (size > cached_size) {
        if (d_prev) cudaFree(d_prev);
        if (d_curr) cudaFree(d_curr);
        if (d_sum) cudaFree(d_sum);
        
        err = cudaMalloc(&d_prev, size);
        if (err != cudaSuccess) {
            d_prev = nullptr; d_curr = nullptr; d_sum = nullptr; cached_size = 0;
            return 0.0f;
        }
        
        err = cudaMalloc(&d_curr, size);
        if (err != cudaSuccess) {
            cudaFree(d_prev);
            d_prev = nullptr; d_curr = nullptr; d_sum = nullptr; cached_size = 0;
            return 0.0f;
        }
        
        err = cudaMalloc(&d_sum, sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(d_prev);
            cudaFree(d_curr);
            d_prev = nullptr; d_curr = nullptr; d_sum = nullptr; cached_size = 0;
            return 0.0f;
        }
        
        cached_size = size;
    }
    
    float initial_sum = 0.0f;
    cudaMemcpy(d_prev, prev_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_curr, curr_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &initial_sum, sizeof(float), cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    motion_magnitude_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(d_prev, d_curr, d_sum, size);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    float host_sum = 0.0f;
    cudaMemcpy(&host_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    return host_sum / size;
}

}
