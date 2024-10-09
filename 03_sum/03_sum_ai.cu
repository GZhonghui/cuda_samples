// 本代码由AI生成，仅供参考
// 已经过测试，结果正确

#include <stdio.h>
#include <cuda_runtime.h>

// 每个线程块中的线程数
#define BLOCK_SIZE 1024

// CUDA内核函数 - 每个块内部的归约求和
__global__ void sum_reduction(float *d_in, float *d_out, int n) {
    extern __shared__ float sdata[];  // 使用共享内存
    
    // 计算全局索引和线程索引
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 将输入数据加载到共享内存
    sdata[tid] = (index < n) ? d_in[index] : 0.0f;
    __syncthreads();  // 确保所有线程加载完成

    // 在共享内存内进行归约求和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();  // 同步，确保每步操作都完成
    }

    // 将每个块的部分和存储到全局内存中
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// 主机函数 - 管理设备内核调用
float parallelSum(float *h_in, int n) {
    float *d_in, *d_out;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 为输入和输出分配设备内存
    cudaMalloc((void **)&d_in, n * sizeof(float));
    cudaMalloc((void **)&d_out, num_blocks * sizeof(float));
    
    // 将数据从主机传输到设备
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    // 计算每个线程块的共享内存大小
    int shared_mem_size = BLOCK_SIZE * sizeof(float);

    // 调用内核函数进行第一次归约
    sum_reduction<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(d_in, d_out, n);

    // 为第二次归约准备，将部分和传回主机
    float *h_out = (float *)malloc(num_blocks * sizeof(float));
    cudaMemcpy(h_out, d_out, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // 主机上执行最终归约
    float total_sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total_sum += h_out[i];
    }

    // 释放内存
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return total_sum;
}

// 主函数
int main() {
    const int n = 1000000;
    float *h_in = (float *)malloc(n * sizeof(float));

    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_in[i] = 1.0f;  // 可改成任意值
    }

    float result = parallelSum(h_in, n);
    printf("Sum: %f\n", result);

    // 释放主机内存
    free(h_in);
    return 0;
}
