//
// Created by hieu on 9/18/25.
//

#ifndef CUDA_STUFFS_KERNEL_CUH
#define CUDA_STUFFS_KERNEL_CUH

#include<cuda_runtime.h>
__global__ void hello_kernel();

auto hello_kernel_cuda() -> void;

auto vec_add_cuda(
	const float *a,
	const float *b,
	float *c,
	size_t n,
	size_t block_size = 256
) -> void;

#endif //CUDA_STUFFS_KERNEL_CUH