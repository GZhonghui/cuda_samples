// 本代码由AI生成，暂时没有经过验证
// 通用的原子操作，不一定是原子加法

#include <cstdio>

__device__ void atomicMul(int* address, int val) {
    int old = *address;  // 读取内存地址上的当前值
    int assumed;         // 这个变量用来暂存推测的旧值

    // 使用循环不断尝试更新值，直到成功为止
    do {
        assumed = old;  // 设置推测的旧值
        // 尝试使用atomicCAS更新内存值：
        // atomicCAS(ptr, compare, val) 的工作原理是：
        // 检查ptr指向的值是否与compare相等，
        // 如果相等，将val写入ptr，并返回旧的值。
        // 如果不相等，返回ptr指向的当前值。
        // 在这里，我们尝试将old值更新为old * val。
        old = atomicCAS(address, assumed, assumed * val);

        // 如果assumed不等于old，说明在执行atomicCAS时，
        // address指向的值已被其他线程修改，因此需要重新尝试。
        // 循环将继续，直到成功为止（即，assumed等于old时退出循环）。
    } while (assumed != old);
}

__global__ void atomicMulExample(int *product) {
    atomicMul(product, 2);  // 试图将product的每个值乘以2
}

int main() {
    int *d_product, h_product = 1;
    
    // 分配设备内存并初始化
    cudaMalloc((void **)&d_product, sizeof(int));
    cudaMemcpy(d_product, &h_product, sizeof(int), cudaMemcpyHostToDevice);
    
    // 定义线程布局：1个block，256个线程
    atomicMulExample<<<1, 256>>>(d_product);
    
    // 等待CUDA核心完成
    cudaDeviceSynchronize();
    
    // 将结果复制回主机
    cudaMemcpy(&h_product, d_product, sizeof(int), cudaMemcpyDeviceToHost);
    
    // 输出结果
    printf("Product = %d\n", h_product);
    
    // 释放设备内存
    cudaFree(d_product);
    
    return 0;
}
