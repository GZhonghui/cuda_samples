#include <algorithm>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <cstdio>

#include <cuda_runtime_api.h>

const int32_t n = 256;

// 每个Thread处理一个数据，所有Thread并行
__global__ void add_on_gpu(int32_t *a, int32_t *b, int32_t *c, int32_t n)
{
    // 0 <= threadIdx.x
    // warning #186-D: pointless comparison of unsigned integer with zero
    if(threadIdx.x < n) {
        c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
    }
}

// 数据太多Thread不够怎么办呢？
// 每个Thread串行处理多个数据也可以的
__global__ void add_on_gpu_any_count(int32_t *a, int32_t *b, int32_t *c, int32_t n)
{
    for(int32_t i = threadIdx.x; i < n; i += blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

inline void print_cuda_error()
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}

inline bool check(int32_t *a, int32_t *b, int32_t *c, int32_t n)
{
    bool found_error = false;
    for(int32_t i = 0; i < n; i += 1) {
        if(c[i] != a[i] + b[i]) {
            found_error = true;

            std::cout << "Error found: " << c[i] << " != " << a[i] << " + " << b[i] << std::endl;

            break;
        }
    }
    if(!found_error) std::cout << "Check passed" << std::endl;

    return !found_error;
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
    print_cuda_error();

    cudaMemcpy(c, c_gpu, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    check(a, b, c, n);

    int32_t t_n = 196; // any count
    add_on_gpu_any_count<<<1,64>>>(a_gpu, b_gpu, c_gpu, t_n);
    cudaDeviceSynchronize();
    print_cuda_error();

    memset(c, 0, sizeof(int32_t) * t_n);
    cudaMemcpy(c, c_gpu, t_n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    check(a, b, c, t_n);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    delete[] a,b,c;
    return 0;
}