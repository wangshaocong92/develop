cmake_minimum_required(VERSION 3.18)

project(kernel LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${GTest_INCLUDE_DIRS}
    .
)

set(DIR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce/reduce.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax/softmax.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/attention/standard_attention/standard_attention.cu
)
file(GLOB DIR_HEADER CONFIGURE_DEPENDS
    "*/*.cuh"
)

message(STATUS "DIR_SRCS: ${DIR_SRCS}")

add_library(kernel SHARED ${DIR_SRCS})

target_link_libraries(kernel CUDA::cudart)

# CUDA 设置
set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/flash_attention/csrc/cutlass/include
)

target_compile_options(kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

enable_testing()
add_subdirectory(test)

install(TARGETS kernel DESTINATION lib)
install(FILES ${DIR_HEADER} DESTINATION include/kernel)







