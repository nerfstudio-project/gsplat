#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>

#include "forward.cuh"
#include "helpers.cuh"

#include "reference/cuda_rasterizer/forward.cu"

float random_float() { return (float)std::rand() / RAND_MAX; }

void compare_cov2d_forward(){
    int num_points = 1;
    float fx = 1;
    float fy = 1;
    const int W = 256;
    const int H = 256;
    float tan_fovx = 0.5 * W / fx;
    float tan_fovy = 0.5 * H / fy;
    float viewmat[] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 8.f,
        0.f, 0.f, 0.f, 1.f};
    float viewmatColumnMajor[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 8.0f, 1.0f
    };
    float cov3d[]{
        1.f, 0.f, 0.f, 
        0.f, 1.f, 0.f,
        0.f, 0.f, 1.f
    };
    float3 mean = {1.f, 1.f, 1.f};

    // diff rast
    float3 diff_rast_cov2d = project_cov3d_ewa(mean,cov3d, viewmat, fx,fy);
    std::cout<<"diff rast cov2d: "<<diff_rast_cov2d.x<<" "<<diff_rast_cov2d.y<<" "<<diff_rast_cov2d.z<<std::endl; 
    
    // ref rast
    float3 res = computeCov2D(mean, fx, fy, tan_fovx, tan_fovy, cov3d, viewmatColumnMajor);
    std::cout<<"ref rast cov2d:  "<<res.x<<" "<<res.y<<" "<<res.z<<std::endl;
}

int main(){
    compare_cov2d_forward();
}