cmake_minimum_required(VERSION 3.18)
project(lbm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)

add_executable(lbm
    src/main.cpp
    src/config.cpp
    src/simulation.cu
    src/boundary.cpp
    src/rendering.cpp
)

target_include_directories(lbm PRIVATE
    src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(lbm PRIVATE
    SFML::Graphics
    SFML::Window
    SFML::System
    CUDA::cudart
)

set_target_properties(lbm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
