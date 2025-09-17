cmake_minimum_required(VERSION 4.0)
project(cuda-stuffs LANGUAGES CXX)
include(cmake/DebugHelper.cmake)

# /--- Setup
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
enable_language(CUDA)

# This will fallback to find FindCudaToolkit.cmake module, not run in config mode.
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA targets founded (All targets that have the namespace prefix 'CUDA::') >>> ")
list_all_cuda_targets()

# /--- Targets
add_executable(main)
target_sources(main PRIVATE src/main.cpp src/kernel.cu)

# /--- Link
#target_link_libraries(main PRIVATE CUDA::cudart)

