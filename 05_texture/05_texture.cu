// 本代码由AI生成，暂时还没有经过检验

#include <cuda_runtime.h>
#include <iostream>

// 定义纹理引用，指定使用float2作为返回类型
// float2表示使用了两个通道（每一个坐标位置都是两个值）
// 纹理引用需要是一个全局变量
texture<float2, cudaTextureType2D, cudaReadModeElementType> texRef;

// 内核函数
__global__ void textureKernel(float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // 使用tex2D从纹理内存中读取两个通道的值
        float2 pixel = tex2D(texRef, x, y);
        // 示例：这里我们将第一个通道的值存储到输出数组中
        output[y * width + x] = pixel.x; // 也可以使用 pixel.y 来访问第二个通道
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    size_t size = width * height * sizeof(float);

    // 主机端数组
    float* hostData = new float[width * height * 2]; // 两个通道
    float* devOutput;

    // 初始化主机端数据
    for (int i = 0; i < width * height * 2; i += 2) {
        hostData[i] = static_cast<float>(i); // 第一个通道
        hostData[i + 1] = static_cast<float>(i + 1); // 第二个通道
    }

    // 创建并配置 channelDesc，为两个浮点通道指定通道格式
    // 两个32位的通道，类型是float，其余可选的还有 无符号整数 和 有符号整数
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

    // 设备端 cudaArray
    // 申请显存，Array 比 普通线性显存 更适合作为texture
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // 将数据从主机复制到设备
    // 参数中的两个0表示w和h方向的偏移
    // Host数据复制到Device，这一点和普通显存一样
    cudaMemcpyToArray(cuArray, 0, 0, hostData, width * height * sizeof(float) * 2, cudaMemcpyHostToDevice);

    // 绑定 cudaArray 到纹理引用
    cudaBindTextureToArray(texRef, cuArray);

    // 分配设备输出缓存
    cudaMalloc(&devOutput, size);

    // 设置网格和块的维度
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    // 启动内核
    textureKernel<<<dimGrid, dimBlock>>>(devOutput, width, height);
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(hostData, devOutput, size, cudaMemcpyDeviceToHost);

    // 清理
    cudaFreeArray(cuArray);
    cudaFree(devOutput);
    delete[] hostData;
    cudaUnbindTexture(texRef);

    return 0;
}
