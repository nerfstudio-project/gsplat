#include "forward.h"
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// host function to launch the projection in parallel on device
void project_gaussians_forward_impl(
    const int num_points,
    const float *means3d,
    const float *scales,
    const float glob_scale,
    const float *rots_quat,
    const float *viewmat,
    float *covs3d,
    float *covs2d
) {
    int num_threads = 16;
    project_gaussians_forward_kernel
    <<< (num_points + num_threads - 1) / num_threads, num_threads >>> (
        num_points,
        means3d,
        scales,
        glob_scale,
        rots_quat,
        viewmat,
        covs3d,
        covs2d
    );
}


// kernel function for projecting each gaussian on device
// each thread processes one gaussian
__global__ void project_gaussians_forward_kernel(
    const int num_points,
    const float *means3d,
    const float *scales,
    const float glob_scale,
    const float *rots_quat,
    const float *viewmat,
    float *covs3d,
    float *covs2d
) {
    unsigned idx = cg::this_grid().thread_rank();  // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    float3 p_world = {means3d[3*idx], means3d[3*idx+1], means3d[3*idx+2]};
    // printf("hello %d", idx);
    // printf("%.2f %.2f %.2f\n", scales[0], scales[1], scales[2]);
    float3 scale = {scales[3*idx], scales[3*idx+1], scales[3*idx+2]};
    float4 quat = {rots_quat[4*idx+1], rots_quat[4*idx +2], rots_quat[4*idx + 3], rots_quat[4*idx]};
    // printf("0 scale %.2f %.2f %.2f\n", scale.x, scale.y, scale.z);
    // printf("0 quat %.2f %.2f %.2f %.2f\n", quat.w, quat.x, quat.y, quat.z);
    compute_cov3d(scale, glob_scale, quat, &(covs3d[6 * idx]));
    project_cov3d_ewa(p_world, &(covs3d[6*idx]), viewmat, 1.f, 1.f, &(covs2d[3*idx]));
}


// host function to launch parallel rendering of sorted gaussians on device
void render_forward_impl(
) {
}


// kernel function for rendering each gaussian on device
__global__ void render_forward_kernel(
) {
}


// device helper to approximate projected 2d cov from 3d mean and cov
__device__ void project_cov3d_ewa(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy,
    // const float tan_fovx,
    // const float tan_fovy,
    float *cov2d
) {
    // we expect row major matrices as input,
    // glm uses column major
    glm::mat4 viewmat_glm = glm::transpose(glm::make_mat4(viewmat));
    glm::vec4 t = viewmat_glm * glm::vec4(mean3d.x, mean3d.y, mean3d.z, 1.f);
    // printf("viewmat_glm %.2f %.2f %.2f %.2f\n", viewmat_glm[0][0], viewmat_glm[1][1], viewmat_glm[2][2], viewmat_glm[3][3]);
    // printf("t %.2f %.2f %.2f %.2f\n", t[0], t[1], t[2], t[3]);
    // printf("t %.2f %.2f %.2f %.2f\n", t.w, t.x, t.y, t.z);

    // column major
    glm::mat3 J = glm::transpose(glm::mat3(
        fx / t.z, 0.f, -fx * t.x / (t.z * t.z),
        0.f, fy / t.z, -fy * t.y / (t.z * t.z),
        0.f, 0.f, 0.f
    ));
    glm::mat3 W = glm::mat3(
        glm::vec3(viewmat_glm[0]), glm::vec3(viewmat_glm[1]), glm::vec3(viewmat_glm[2])
    );

    glm::mat3 T = J * W;

    glm::mat3 V = glm::mat3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );

    // printf("J %.2f %.2f %.2f\n", J[0][0], J[1][1], J[2][2]);
    // printf("W %.2f %.2f %.2f\n", W[0][0], W[1][1], W[2][2]);
    // printf("V %.2f %.2f %.2f\n", V[0][0], V[1][1], V[2][2]);

    // we only care about the top 2x2 submatrix
    glm::mat3 cov = T * V * glm::transpose(T);
    // add a little blur along axes and save upper triangular elements
    cov2d[0] = float(cov[0][0]) + 0.1f;
    cov2d[1] = float(cov[0][1]);
    cov2d[2] = float(cov[1][1]) + 0.1f;
}


// device helper to get 3D covariance from scale and quat parameters
__device__ void compute_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
) {
    // printf("scale %.2f %.2f %.2f\n", scale.x, scale.y, scale.z);
    // printf("quat %.2f %.2f %.2f %.2f\n", quat.w, quat.x, quat.y, quat.z);
    // quat to rotation matrix
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.w * s;
    float x = quat.x * s;
    float y = quat.y * s;
    float z = quat.z * s;

    // glm matrices are column-major
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
        2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
        2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + x * x)
    );
    // printf("R %.2f %.2f %.2f\n", R[0][0], R[1][1], R[2][2]);

    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    // printf("S %.2f %.2f %.2f\n", S[0][0], S[1][1], S[2][2]);

    glm::mat3 M = R * S;
    glm::mat tmp = M * glm::transpose(M);
    // printf("tmp %.2f %.2f %.2f\n", tmp[0][0], tmp[1][1], tmp[2][2]);

    // save upper right because symmetric
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}
