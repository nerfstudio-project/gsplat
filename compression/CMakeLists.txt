cmake_minimum_required(VERSION 3.18)
project(PointCloudCompression LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set installation prefixes (matching install_dependencies.sh)
set(PCL_VERSION "1.15.1" CACHE STRING "PCL version")
set(DRACO_VERSION "1.5.7" CACHE STRING "Draco version")
set(OPEN3D_VERSION "v0.19.0" CACHE STRING "Open3D version")

set(PCL_ROOT "$ENV{HOME}/tools/pcl-${PCL_VERSION}/install" CACHE PATH "PCL installation directory")
set(DRACO_ROOT "$ENV{HOME}/tools/draco-${DRACO_VERSION}/install" CACHE PATH "Draco installation directory")
set(Open3D_ROOT "$ENV{HOME}/tools/open3d-${OPEN3D_VERSION}/install" CACHE PATH "Open3D installation directory")

# Add custom paths to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH ${PCL_ROOT} ${DRACO_ROOT} ${Open3D_ROOT})

# Find PCL
find_package(PCL REQUIRED)
if(PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    message(STATUS "PCL found: ${PCL_VERSION} at ${PCL_ROOT}")
else()
    message(FATAL_ERROR "PCL not found. Please install PCL or set PCL_ROOT")
endif()

# Find Open3D
find_package(Open3D REQUIRED)
if(Open3D_FOUND)
    message(STATUS "Open3D found at ${Open3D_ROOT}")
else()
    message(FATAL_ERROR "Open3D not found. Please install Open3D or set Open3D_ROOT")
endif()

# Find Draco
find_package(draco REQUIRED)
if(draco_FOUND)
    message(STATUS "Draco found at ${DRACO_ROOT}")
else()
    message(FATAL_ERROR "Draco not found. Please install Draco or set DRACO_ROOT")
endif()

# Find jsoncpp (required for Open3D octree JSON serialization)
find_package(PkgConfig REQUIRED)
if(NOT PkgConfig_FOUND)
    message(FATAL_ERROR "PkgConfig not found. Please install PkgConfig")
endif()
pkg_check_modules(JSONCPP jsoncpp REQUIRED)
if(JSONCPP_FOUND)
    message(STATUS "jsoncpp found: version ${JSONCPP_VERSION}")
    message(STATUS "jsoncpp libraries: ${JSONCPP_LIBRARIES}")
    message(STATUS "jsoncpp include dirs: ${JSONCPP_INCLUDE_DIRS}")
    include_directories(${JSONCPP_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "jsoncpp not found. Please install jsoncpp")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/pcl_compression.cpp
    src/open3d_compression.cpp
    src/draco_compression.cpp
    src/gpu_octree_compression.cpp
    src/utils.cpp
    src/timer.cpp
)

# Create executable
add_executable(pointcloud_compression ${SOURCES})

# Set gpu_octree_compression.cpp to compile as CUDA code
set_source_files_properties(src/gpu_octree_compression.cpp PROPERTIES
    LANGUAGE CUDA
)

# Set CUDA properties for pointcloud_compression
set_target_properties(pointcloud_compression PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
)

# Set CUDA include directories
set(CUDA_ROOT "/home/rajrup/cuda-11.8")
target_include_directories(pointcloud_compression PRIVATE
    ${CUDA_ROOT}/include
)

# Find and link CUDA runtime library
find_library(CUDA_CUDART cudart
    PATHS
    ${CUDA_ROOT}/lib64
    ${CUDA_ROOT}/lib
    NO_DEFAULT_PATH
)

if(CUDA_CUDART)
    target_link_libraries(pointcloud_compression ${CUDA_CUDART})
    message(STATUS "CUDA runtime library found: ${CUDA_CUDART}")
endif()

# Link libraries
target_link_libraries(pointcloud_compression 
    ${PCL_LIBRARIES}
    Open3D::Open3D
    draco::draco
    ${JSONCPP_LIBRARIES}
    ffi
)

# GPU Octree executable (requires CUDA)
add_executable(gpu_octree_compression src/gpu_octree.cpp)

# Set the source file to compile as CUDA code
set_source_files_properties(src/gpu_octree.cpp PROPERTIES
    LANGUAGE CUDA
)

# Set CUDA properties
set_target_properties(gpu_octree_compression PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
)

# Set CUDA include directories (CUDA is at /home/rajrup/cuda-11.8)
set(CUDA_ROOT "/home/rajrup/cuda-11.8")
target_include_directories(gpu_octree_compression PRIVATE
    ${CUDA_ROOT}/include
)

# Find and link CUDA runtime library
find_library(CUDA_CUDART cudart
    PATHS
    ${CUDA_ROOT}/lib64
    ${CUDA_ROOT}/lib
    NO_DEFAULT_PATH
)

if(CUDA_CUDART)
    target_link_libraries(gpu_octree_compression ${CUDA_CUDART})
    message(STATUS "CUDA runtime library found: ${CUDA_CUDART}")
else()
    message(WARNING "CUDA runtime library not found at ${CUDA_ROOT}")
endif()

target_link_libraries(gpu_octree_compression 
    ${PCL_LIBRARIES}
    Open3D::Open3D
    draco::draco
    ${JSONCPP_LIBRARIES}
    ffi
)

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(pointcloud_compression PRIVATE -Wall -Wextra)
endif()
