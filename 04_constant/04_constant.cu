#include <algorithm>
#include <iostream>

const int maxn = 32;

float fib[maxn];
__constant__ float gpu_fib[maxn]; // 声明GPU常量

__global__ void f(float *res, int n)
{
    // if(threadIdx.x < maxn)
    // 这样写好像也对（Kernel内部访问Host端的const常量，这个应该是放在内存中的吧，不在显存中），速度会有什么差异吗？

    if(threadIdx.x < n)
    {
        res[threadIdx.x] = gpu_fib[threadIdx.x] + 1; // 访问GPU常量
    }
}

int main()
{
    fib[0] = 0, fib[1] = 1;
    for(int i = 2; i < maxn; i += 1) fib[i] = fib[i - 1] + fib[i - 2];

    // 将常量数据从Host拷贝到Device
    // GPU常量不需要显式Malloc和Free
    cudaMemcpyToSymbol(gpu_fib, fib, sizeof(float) * maxn);

    float res[maxn], *gpu_res;
    cudaMalloc(&gpu_res, sizeof(float) * maxn);

    f<<<1,maxn>>>(gpu_res, maxn);
    cudaDeviceSynchronize();

    cudaMemcpy(res, gpu_res, sizeof(float) * maxn, cudaMemcpyDeviceToHost);
    cudaFree(gpu_res);

    bool found_error = false;
    for(int i = 0; i < maxn; i += 1)
    {
        found_error |= (res[i] != fib[i] + 1);
    }
    std::cout << (found_error ? "found error" : "check passed") << std::endl;

    return 0;
}