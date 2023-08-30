cmake_minimum_required(VERSION 3.12)  # You can adjust the minimum required version

project(diff_rast CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Add library
add_library(helpers forward.cu)
target_link_libraries(helpers PUBLIC cuda)
target_include_directories(helpers PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/glm
)

# Add the executable
# add_executable(test main.cpp tgaimage.h tgaimage.cpp)
add_executable(test_cuda main.cu)

# Link against CUDA runtime library
target_link_libraries(test_cuda PUBLIC cuda helpers)

# Include directories for the header-only library
target_include_directories(test_cuda PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/glm
)
