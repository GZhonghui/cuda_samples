// 本代码由AI生成，暂时没有经过验证

#include <iostream>
#include <cuda_runtime.h>

// 一个简单的CUDA内核，将数组的每个元素加1
__global__ void addOne(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1;
    }
}

int main() {
    int N = 1024 * 1024;  // 数组大小
    int *d_A, *d_B;
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));

    // 创建两个流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 初始化设备内存
    cudaMemset(d_A, 0, N * sizeof(int));
    cudaMemset(d_B, 0, N * sizeof(int));

    // 配置执行参数
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // 在流1中启动内核
    addOne<<<numBlocks, blockSize, 0, stream1>>>(d_A, N);
    
    // 在流2中启动内核
    addOne<<<numBlocks, blockSize, 0, stream2>>>(d_B, N);

    // 等待流完成
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 清理资源
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_A);
    cudaFree(d_B);

    std::cout << "Completed successfully!\n";
    return 0;
}
