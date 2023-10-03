cmake_minimum_required(VERSION 3.12)  # You can adjust the minimum required version
set(CMAKE_CUDA_ARCHITECTURES 70 75 89)  # Ti 2080 uses 75. V100 uses 70. RTX 4090 uses 89.

project(diff_rast CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# our library library
add_library(diffrast forward.cu backward.cu helpers.cuh serial_backward.cu)
target_link_libraries(diffrast PUBLIC cuda)
target_include_directories(diffrast PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/glm
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
set_target_properties(diffrast PROPERTIES CUDA_ARCHITECTURES "70;75;86")

# their library
set(REF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/reference/cuda_rasterizer)
add_library(CudaRasterizer
    ${REF_SOURCE_DIR}/backward.h
    ${REF_SOURCE_DIR}/backward.cu
    ${REF_SOURCE_DIR}/forward.h
    ${REF_SOURCE_DIR}/forward.cu
    ${REF_SOURCE_DIR}/auxiliary.h
    ${REF_SOURCE_DIR}/rasterizer_impl.cu
    ${REF_SOURCE_DIR}/rasterizer_impl.h
    ${REF_SOURCE_DIR}/rasterizer.h
)

target_include_directories(CudaRasterizer PUBLIC ${REF_SOURCE_DIR})
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")

# Add the executable
# add_executable(test main.cpp tgaimage.h tgaimage.cpp)
add_executable(test_forward test_forward.cu tgaimage.cpp)
add_executable(check_serial_backward check_serial_backward.cu debug_utils.cpp)
add_executable(check_serial_forward check_serial_forward.cu forward.cu debug_utils.cpp)

# Link against CUDA runtime library
target_link_libraries(test_forward PUBLIC cuda diffrast)
target_link_libraries(check_serial_backward PUBLIC cuda diffrast)
target_link_libraries(check_serial_forward PUBLIC cuda diffrast CudaRasterizer)

# Include directories for the header-only library
target_include_directories(test_forward PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/glm
)
target_include_directories(check_serial_backward PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/glm
)
target_include_directories(check_serial_forward PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/glm
)