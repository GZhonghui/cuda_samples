#include <cstdio>
#include <iostream>

int main() {

    int nDevices;
    cudaGetDeviceCount(&nDevices);
  
    printf("Number of devices: %d\n", nDevices);
  
    for (int i = 0; i < nDevices; i++) {
        // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // https://stackoverflow.com/questions/14800009/how-to-get-properties-from-active-cuda-device
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n",
            prop.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
        // END stackoverflow

        std::cout << "  maxGridSize = (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << ")" << std::endl;
        
        std::cout << "  maxThreadsPerBlock = " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  maxThreadsDim = (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << ")" << std::endl;

        std::cout << "  sharedMemPerBlock = " << prop.sharedMemPerBlock << " byte(s)" << std::endl;
    }

    return 0;
}