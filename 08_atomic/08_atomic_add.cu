// 本代码由AI生成，暂时没有经过验证

#include <stdio.h>

__global__ void atomicAddExample(int *sum) {
    // 每个线程将1累加到共享变量sum上
    atomicAdd(sum, 1);
}

int main() {
    int *d_sum, h_sum = 0;
    
    // 分配设备内存
    cudaMalloc((void **)&d_sum, sizeof(int));
    
    // 初始化sum为0
    cudaMemcpy(d_sum, &h_sum, sizeof(int), cudaMemcpyHostToDevice);
    
    // 定义线程布局：1个block，256个线程
    int numThreads = 256;
    atomicAddExample<<<1, numThreads>>>(d_sum);
    
    // 等待CUDA核心完成
    cudaDeviceSynchronize();
    
    // 将结果复制回主机
    cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    
    // 输出结果
    printf("Sum = %d\n", h_sum);
    
    // 释放设备内存
    cudaFree(d_sum);
    
    return 0;
}
