cmake_minimum_required(VERSION 4.0)

project(cuda-stuffs LANGUAGES CXX)
include(cmake/DebugHelper.cmake)

# /--- Setup CUDA
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
set(CMAKE_CUDA_SEPARABLE_COMPILATION on)
enable_language(CUDA)

# This will fallback to find FindCudaToolkit.cmake module, not run in config
# mode.
find_package(CUDAToolkit REQUIRED)
message(
  STATUS
    "CUDA targets founded (All targets that have the namespace prefix 'CUDA::') >>> "
)
list_all_cuda_targets()

# /--- Targets
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*")
list(FILTER SOURCES INCLUDE REGEX ".*\\.(cpp|cxx|cc|c|cu)$")

add_executable(main)
target_sources(main PRIVATE ${SOURCES})
target_compile_features(main PRIVATE cxx_std_23)
target_include_directories(main PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
#target_link_libraries(main PRIVATE CUDA::cudart)
