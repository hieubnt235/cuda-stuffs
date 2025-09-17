//
// Created by hieu on 9/16/25.
//

#include <iostream>
#include <cuda_runtime.h>
#include<thrust/device_vector.h>

__global__ void hello_kernel() {
    printf("Hello World from GPU!\n");
}

int main() {
    // Launch kernel with 1 block, 1 thread
    hello_kernel<<<1, 1>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    std::cout << "Hello World from CPU!" << std::endl;
    return 0;
}
