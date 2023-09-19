#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>

#include "forward.cuh"
#include "helpers.cuh"

#include "debug_utils.h"

#include "reference/cuda_rasterizer/forward.cu"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

float random_float() { return (float)std::rand() / RAND_MAX; }

float4 random_quat() {
    float u = random_float();
    float v = random_float();
    float w = random_float();
    return {
        sqrt(1.f - u) * sin(2.f * (float)M_PI * v),
        sqrt(1.f - u) * cos(2.f * (float)M_PI * v),
        sqrt(u) * sin(2.f * (float)M_PI * w),
        sqrt(u) * cos(2.f * (float)M_PI * w)};
}

void compare_cov2d_forward(){
    // TODO: test with more than one point and varying cov3d/viewmat
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
        1.f, 0.f, 1.f
    };
    float3 mean = {1.f, 1.f, 1.f};

    // diff rast
    float3 diff_rast_cov2d = project_cov3d_ewa(mean,cov3d, viewmat, fx,fy);
    std::cout<<"diff rast cov2d: "<<diff_rast_cov2d.x<<" "<<diff_rast_cov2d.y<<" "<<diff_rast_cov2d.z<<std::endl; 
    
    // ref rast
    float3 res = computeCov2D(mean, fx, fy, tan_fovx, tan_fovy, cov3d, viewmatColumnMajor);
    std::cout<<"ref rast cov2d:  "<<res.x<<" "<<res.y<<" "<<res.z<<std::endl;


    const std::vector<std::pair<float,float>> cov2d_data {
        {diff_rast_cov2d.x, res.x},
        {diff_rast_cov2d.y, res.y},
        {diff_rast_cov2d.z, res.z},
    };
    print_errors(cov2d_data, "diff_rast_cov2d (cov2d_forward)");
}

void compare_scale_rot_to_cov3d(){
    // TODO: make it work with more than one point
    int num_points = 1;
    float3 scale = {random_float(), random_float(), random_float()};
    float glob_scale = random_float();

    float4 quat = random_quat();

    float* cov3d = new float[6 * num_points];

    // diff rast
    scale_rot_to_cov3d(scale, glob_scale, quat, cov3d);   
    std::cout << "diff rast cov3d: " << cov3d[0]<<" "<<cov3d[1]<<" "<<cov3d[2]<<std::endl;

    // ref rast
    float* ref_cov3d = new float[6 * num_points];
    computeCov3D(glm::vec3({scale.x,scale.y,scale.z}), glob_scale, glm::vec4({quat.x, quat.y, quat.z, quat.w}), ref_cov3d);
    std::cout << "ref rast cov3d: " <<ref_cov3d[0]<<" "<<ref_cov3d[1]<<" "<<ref_cov3d[2]<<std::endl;


    std::vector<std::pair<float,float>> cov3d_data;
    for (int i = 0; i < 3; ++i)   // Jonathan: is this supposed to just go to 3?
        cov3d_data.push_back({cov3d[i], ref_cov3d[i]});

    print_errors(cov3d_data, "diff rast cov3d (scale_rot_to_cov3d)");
}

// doctest example code start
int factorial(int number) { return number <= 1 ? number : factorial(number - 1) * number; }

TEST_CASE("testing the factorial function") {
    CHECK(factorial(1) == 1);
    CHECK(factorial(2) == 2);
    CHECK(factorial(3) == 6);
    CHECK(factorial(10) == 3628800);
    CHECK(factorial(0) == -123);    // intentional failure; remove me to see
                                    // what passing tests looks like :)
}
// doctest example code end

int main(){

    compare_cov2d_forward();
    compare_scale_rot_to_cov3d();

    // todo: write doctest tests
    doctest::Context context;

    int results = context.run();    // runs tests

    return results;
}
