//
// Created by hieu on 9/16/25.
//

#include <cmath>
#include<print>
#include <ranges>

#include "kernel.cuh"

int main() {
	hello_kernel_cuda();
	std::print("Hello World from CPU!\n");

	constexpr size_t N = 1000;
	std::array<float, N> A{};
	std::array<float, N> B{};
	std::array<float, N> C{};

	for (size_t i{0}; i != N; ++i) {
		A[i] = static_cast<float>(i);
		B[i] = static_cast<float>(std::pow(i, 2));
	}
	vec_add_cuda(A.data(), B.data(), C.data(), N);

	for (auto [a,b,c] : std::views::zip(A, B, C)) {
		std::print("{}+{}={}\n", a, b, c);
	}

	return 0;
}