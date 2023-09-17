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

__global__ void cov2d_kernel_entry(const float3& mean, float fx, float fy, float tan_fovx, float tan_fovy, const float* cov3d, const float* viewmat, float3* ref_res_d, const float num_points)
{
    // entry point to call reference device code
    auto idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
		return;

    float3 res = computeCov2D(mean, fx, fy, tan_fovx, tan_fovy, cov3d, viewmat);
    ref_res_d[idx] = res;
    
}
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
    float3 *ref_res_d;// = new float3[num_points];
    cudaMalloc((void **)&ref_res_d, num_points * sizeof(float3));

    cov2d_kernel_entry<<<1,1>>>(mean, fx, fy, tan_fovx, tan_fovy, cov3d, viewmatColumnMajor, ref_res_d, num_points);
    cudaDeviceSynchronize();

    float3 ref_rast_cov2d[num_points];
    cudaMemcpy(
        ref_rast_cov2d, ref_res_d, num_points * sizeof(float3), cudaMemcpyDeviceToHost
    );

    cudaFree(ref_res_d);

    // Why is the output not correct :(
    std::cout<<"ref rast cov2d:  "<<ref_rast_cov2d[0].x<<" "<<ref_rast_cov2d[0].y<<" "<<ref_rast_cov2d[0].z<<std::endl;
}

int main(){
    compare_cov2d_forward();
}