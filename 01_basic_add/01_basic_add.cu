#include <algorithm>
#include <iostream>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime_api.h>

const int32_t n = 256;

__global__ void add_on_gpu(int32_t *a, int32_t *b, int32_t *c, int32_t n)
{
    // 0 <= threadIdx.x
    // warning #186-D: pointless comparison of unsigned integer with zero
    if(threadIdx.x < n) {
        c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
    }
}

int main()
{
    int32_t *a = new int32_t[n], *b = new int32_t[n], *c = new int32_t[n];

    for(int32_t i = 0; i < n; i += 1) {
        a[i] = i * i;
        b[i] = i * i * i;
    }

    int32_t *a_gpu, *b_gpu, *c_gpu;
    cudaMalloc(&a_gpu, n * sizeof(int32_t));
    cudaMalloc(&b_gpu, n * sizeof(int32_t));
    cudaMalloc(&c_gpu, n * sizeof(int32_t));

    cudaMemcpy(a_gpu, a, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, n * sizeof(int32_t), cudaMemcpyHostToDevice);

    add_on_gpu<<<1,n>>>(a_gpu, b_gpu, c_gpu, n);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }

    cudaMemcpy(c, c_gpu, n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    bool found_error = false;
    for(int32_t i = 0; i < n; i += 1) {
        if(c[i] != a[i] + b[i]) {
            found_error = true;

            std::cout << "Error found: " << c[i] << " != " << a[i] << " + " << b[i] << std::endl;

            break;
        }
    }
    if(!found_error) std::cout << "Check passed" << std::endl;

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    delete[] a,b,c;
    return 0;
}