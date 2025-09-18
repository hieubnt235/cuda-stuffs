//
// Created by hieu on 9/16/25.
//

// #include<iostream>
#include"kernel.cuh"

#include <cmath>
#include <format>
#include<iostream>
__global__ void hello_kernel() {
	printf("Hello World from GPU!\n");
}

__global__ auto vec_add_kernel(
	const float *a,
	const float *b,
	float *c,
	const size_t n
) -> void {
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n)
		c[i] = a[i] + b[i];
}

auto vec_add_cuda(
	const float *a,
	const float *b,
	float *c,
	const size_t n,
	size_t block_size
) -> void {
	float *a_d{nullptr}, *b_d{nullptr}, *c_d{nullptr};
	size_t size = sizeof(float) * n;

	auto malloc_device = [size](float **v)-> void {
		cudaMalloc(reinterpret_cast<void **>(v), size);
	};
	malloc_device(&a_d);
	malloc_device(&b_d);
	malloc_device(&c_d);

	cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);

	// Fill n into block_size, return number of block that needed to contain all n.
	auto resolve = [](size_t n, size_t block_size)-> size_t {
		return static_cast<size_t>(std::ceil(
				static_cast<double>(n) / static_cast<double>(block_size)
			)
		);
	};

	block_size = resolve(block_size, 32)*32;
	auto grid_size = resolve(n, block_size);

	std::cout << std::format(
		"Calling vec_add_kernel with <<<{},{}>>>",
		grid_size,
		block_size
	) << std::endl;
	vec_add_kernel<<<grid_size, block_size>>>(
		a_d,
		b_d,
		c_d,
		n
	);
	cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost);
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
}

void hello_kernel_cuda() {
	int count;
	cudaError_t err = cudaGetDeviceCount(&count);
	printf("GPU count = %d\n", count);

	hello_kernel<<<1,1>>>();
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		printf("CUDA error: %s\n", cudaGetErrorString(err));
}