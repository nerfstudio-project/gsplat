#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <math.h>

#include "forward.cuh"
#include "backward.cuh"
#include "backward_ref.cuh"
#include "helpers.cuh"


float random_float() {
    return (float) std::rand() / RAND_MAX;
}

float4 random_quat() {
    float u = random_float();
    float v = random_float();
    float w = random_float();
    return {
        sqrt(1.f - u) * sin(2.f * (float) M_PI * v),
        sqrt(1.f - u) * cos(2.f * (float) M_PI * v),
        sqrt(u) * sin(2.f * (float) M_PI * w),
        sqrt(u) * cos(2.f * (float) M_PI * w)
    };
}

float3 compute_conic(const float3 cov2d) {
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return {0.f, 0.f, 0.f};
    float inv_det = 1.f / det;
    float3 conic;
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;
    return conic;
}

void compare_project2d_mean_backward() {
    float3 mean = {random_float(), random_float(), random_float()};
    float proj[] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    };
    float2 dL_dmean2d = {random_float(), random_float()};

    float3 dL_dmean = project_pix_vjp(proj, mean, (dim3) {1, 1, 1}, dL_dmean2d);
    float3 dL_dmean_ref = projectMean2DBackward(mean, proj, dL_dmean2d);
    printf("project2d backward\n");
    printf("dL_dmean\n");
    printf("ours %.2e %.2e %.2e\n", dL_dmean.x, dL_dmean.y, dL_dmean.z);
    printf("ref %.2e %.2e %.2e\n", dL_dmean_ref.x, dL_dmean_ref.y, dL_dmean_ref.z);
}

void compare_conic_backward() {
    float3 cov2d = {random_float(), random_float(), random_float()};
    float3 conic = compute_conic(cov2d);
    float3 dL_dconic = {random_float(), random_float(), random_float()};
    float3 dL_dcov2d;
    cov2d_to_conic_vjp(conic, dL_dconic, dL_dcov2d);
    float3 dL_dcov2d_ref = computeConicBackward(cov2d, dL_dconic);

    printf("conic backward\n");
    printf("dL_dcov2d\n");
    printf("ours %.2e %.2e %.2e\n", dL_dcov2d.x, dL_dcov2d.y, dL_dcov2d.z);
    printf("ref %.2e %.2e %.2e\n", dL_dcov2d_ref.x, dL_dcov2d_ref.y, dL_dcov2d_ref.z);

}

void compare_cov3d_backward() {
    float3 scale = {random_float(), random_float(), random_float()};
    float4 quat = random_quat();
    float4 quat_ref = {quat.w, quat.x, quat.y, quat.z};
    float dL_dcov3d [] = {random_float(), random_float(), random_float(), random_float(), random_float(), random_float()};
    float3 dL_ds = {0.f, 0.f, 0.f};
    float4 dL_dq = {0.f, 0.f, 0.f, 0.f};
    scale_rot_to_cov3d_vjp(scale, 1.f, quat, dL_dcov3d, dL_ds, dL_dq);

    float3 dL_ds_ref = {0.f, 0.f, 0.f};
    float4 dL_dq_ref = {0.f, 0.f, 0.f, 0.f};
    computeCov3DBackward(scale, 1.f, quat_ref, dL_dcov3d, dL_ds_ref, dL_dq_ref);

    printf("cov3d backward\n");
    printf("dL_dscale\n");
    printf("ours %.2e %.2e %.2e\n", dL_ds.x, dL_ds.y, dL_ds.z);
    printf("ref %.2e %.2e %.2e\n", dL_ds_ref.x, dL_ds_ref.y, dL_ds_ref.z);

    printf("dL_dquat\n");
    printf("ours %.2e %.2e %.2e %.2e\n", dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w);
    printf("ref %.2e %.2e %.2e %.2e\n", dL_dq_ref.y, dL_dq_ref.z, dL_dq_ref.w, dL_dq_ref.x);
}

void compare_cov2d_ewa_backward() {
    float3 mean = {random_float(), random_float(), random_float()};
    float3 scale = {random_float(), random_float(), random_float()};
    float4 quat = random_quat();
    float cov3d [] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    scale_rot_to_cov3d(scale, 1.f, quat, cov3d);
    float3 dL_dcov2d = {random_float(), random_float(), random_float()};
    float3 dL_dmean, dL_dmean_ref;
    float dL_dcov [] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float dL_dcov_ref [] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // functions expect different view matrix convention
    float viewmat [] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 10.f,
        0.f, 0.f, 0.f, 1.f
    };
    float viewmat_ref [] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 10.f, 1.f
    };
    computeCov2DBackward(mean, cov3d, viewmat_ref, 1.f, 1.f, 1.f, 1.f, dL_dcov2d, dL_dmean_ref, dL_dcov_ref);
    project_cov3d_ewa_vjp(mean, cov3d, viewmat, 1.f, 1.f, dL_dcov2d, dL_dmean, dL_dcov);
    printf("cov2d_ewa backward\n");
    printf("dL_dmean\n");
    printf("ours %.2e %.2e %.2e\n", dL_dmean.x, dL_dmean.y, dL_dmean.z);
    printf("ref %.2e %.2e %.2e\n", dL_dmean_ref.x, dL_dmean_ref.y, dL_dmean_ref.z);
    printf("dL_dcov\nours ");
    for (int i = 0; i < 6; ++i) {
        printf("%.2e ", dL_dcov[i]);
    }
    printf("\ntheirs ");
    for (int i = 0; i < 6; ++i) {
        printf("%.2e ", dL_dcov[i]);
    }
    printf("\n");
}

void compare_rasterize_backward() {
    int N = 16;
}

int main() {
    compare_project2d_mean_backward();
    compare_conic_backward();
    compare_cov3d_backward();
    compare_cov2d_ewa_backward();
    return 0;
}
