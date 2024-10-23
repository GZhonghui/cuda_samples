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
    int *h_A, *d_A;

    // 在主机端分配内存并初始化数据
    h_A = new int[N];
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
    }

    // 在设备端分配内存
    cudaMalloc(&d_A, N * sizeof(int));

    // 创建一个流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 将数据从主机复制到设备
    cudaMemcpyAsync(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice, stream);

    // 配置执行参数
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // 在流中启动内核
    addOne<<<numBlocks, blockSize, 0, stream>>>(d_A, N);

    // 将数据从设备复制回主机
    cudaMemcpyAsync(h_A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost, stream);

    // 等待流完成所有操作
    cudaStreamSynchronize(stream);

    // 输出一些结果进行验证
    std::cout << "First elements: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_A[i] << " ";
    }
    std::cout << "\n";

    // 清理资源
    cudaStreamDestroy(stream);
    cudaFree(d_A);
    delete[] h_A;

    std::cout << "Completed successfully!\n";
    return 0;
}
