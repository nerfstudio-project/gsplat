cmake_minimum_required(VERSION 3.12)  # You can adjust the minimum required version

project(diff_rast CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Add library
add_library(librast forward.cu backward.cu helpers.cuh)
target_link_libraries(librast PUBLIC cuda)
target_include_directories(librast PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/glm
)

# Add the executable
# add_executable(test main.cpp tgaimage.h tgaimage.cpp)
add_executable(test_forward test_forward.cu tgaimage.cpp)
add_executable(test_backward test_backward.cu tgaimage.cpp)

# Link against CUDA runtime library
target_link_libraries(test_forward PUBLIC cuda librast)
target_link_libraries(test_backward PUBLIC cuda librast)

# Include directories for the header-only library
target_include_directories(test_forward PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/glm
)
target_include_directories(test_backward PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/glm
)
