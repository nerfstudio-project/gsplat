cmake_minimum_required(VERSION 3.12)  # You can adjust the minimum required version

project(diff_rast)

# Add the executable
add_executable(test main.cpp)

# Include directories for the header-only library
target_include_directories(test PRIVATE ${PROJECT_SOURCE_DIR}/third_party/glm)
