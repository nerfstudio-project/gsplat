#include "forward.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


// host function to launch the projection in parallel on device
int project_gaussians_forward_impl(
) {
}


// kernel function for projecting each gaussian on device
__global__ int project_gaussians_forward_kernel(
) {
}


// host function to launch parallel rendering of sorted gaussians on device
int render_forward_impl(
) {
}


// kernel function for rendering each gaussian on device
__global__ int render_forward_kernel(
) {
}


// device helper to approximate projected 2d cov from 3d mean and cov
__device__ float3 project_cov3d_ewa(
    const float3 &mean3d, const float *cov3d,
    const float *view_mat_ptr,
    const float fx, fx, tan_fovx, tan_fovy
) {
    // we expect row major matrices as input,
    // glm uses column major
    glm::mat4 view_mat = glm::transpose(glm::make_mat4(view_mat_ptr));
    glm::vec4 t = view_mat * glm::vec4(mean3d.x, mean3d.y, mean3d.z, 1.f);

    // column major
    glm::mat3 J = glm::mat3(
        fx / t.z, 0.f, 0.f,
        0.f, fy / t.z, 0.f,
        -fx * t.x / (t.z * t.z), -fy * t.y / (t.z * t.z), 0.f
    );
    glm::mat3 W = glm::mat3(
        glm::vec3(view_mat[0]), glm::vec3(view_mat[1]), glm::vec3(view_mat[2])
    );
    glm::mat3 T = J * W

    glm::mat3 V = glm::mat3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );

    // we only care about the top 2x2 submatrix
    glm::mat3 cov = T * V * glm::transpose(T);
    // add a little blur along axes
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
    // return the upper triangular elements
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}


// device helper to get 3D covariance from scale and quat parameters
__device__ void compute_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, const float *cov3d
) {
    // quat to rotation matrix
    float s = rsqrtf(
        quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]
    );
    float w = quat[0] * s;
    float x = quat[1] * s;
    float y = quat[2] * s;
    float z = quat[3] * s;

    // glm matrices are column-major
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
        2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
        2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + x * x)
    );

    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = glob_scale * scale[0];
    S[1][1] = glob_scale * scale[1];
    S[2][2] = glob_scale * scale[2];

    glm::mat3 M = R * S;
    glm::mat tmp = M * glm::transpose(M);

    // save upper right because symmetric
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}
