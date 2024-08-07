#ifndef GSPLAT_CUDA_UTILS_H
#define GSPLAT_CUDA_UTILS_H

#include "helpers.cuh"
#include "symeigen.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

template <typename T> inline __device__ mat3<T> quat_to_rotmat(const vec4<T> quat) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    T inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    T x2 = x * x, y2 = y * y, z2 = z * z;
    T xy = x * y, xz = x * z, yz = y * z;
    T wx = w * x, wy = w * y, wz = w * z;
    return mat3<T>((1.f - 2.f * (y2 + z2)), (2.f * (xy + wz)),
                   (2.f * (xz - wy)), // 1st col
                   (2.f * (xy - wz)), (1.f - 2.f * (x2 + z2)),
                   (2.f * (yz + wx)), // 2nd col
                   (2.f * (xz + wy)), (2.f * (yz - wx)),
                   (1.f - 2.f * (x2 + y2)) // 3rd col
    );
}

template <typename T>
inline __device__ void quat_to_rotmat_vjp(const vec4<T> quat, const mat3<T> v_R,
                                          vec4<T> &v_quat) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    T inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    vec4<T> v_quat_n = vec4<T>(
        2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
               z * (v_R[0][1] - v_R[1][0])),
        2.f * (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
               z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])),
        2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
               z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])),
        2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
               2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])));

    vec4<T> quat_n = vec4<T>(w, x, y, z);
    v_quat += (v_quat_n - glm::dot(v_quat_n, quat_n) * quat_n) * inv_norm;
}

template <typename T>
inline __device__ void quat_scale_to_covar_preci(const vec4<T> quat,
                                                 const vec3<T> scale,
                                                 // optional outputs
                                                 mat3<T> *covar, mat3<T> *preci) {
    mat3<T> R = quat_to_rotmat<T>(quat);
    if (covar != nullptr) {
        // C = R * S * S * Rt
        mat3<T> S = mat3<T>(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
        mat3<T> M = R * S;
        *covar = M * glm::transpose(M);
    }
    if (preci != nullptr) {
        // P = R * S^-1 * S^-1 * Rt
        mat3<T> S = mat3<T>(1.0f / scale[0], 0.f, 0.f, 0.f, 1.0f / scale[1], 0.f, 0.f,
                            0.f, 1.0f / scale[2]);
        mat3<T> M = R * S;
        *preci = M * glm::transpose(M);
    }
}

template <typename T>
inline __device__ void quat_scale_to_covar_vjp(
    // fwd inputs
    const vec4<T> quat, const vec3<T> scale,
    // precompute
    const mat3<T> R,
    // grad outputs
    const mat3<T> v_covar,
    // grad inputs
    vec4<T> &v_quat, vec3<T> &v_scale) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    T sx = scale[0], sy = scale[1], sz = scale[2];

    // M = R * S
    mat3<T> S = mat3<T>(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3<T> M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3<T> v_M = (v_covar + glm::transpose(v_covar)) * M;
    mat3<T> v_R = v_M * S;

    // grad for (quat, scale) from covar
    quat_to_rotmat_vjp<T>(quat, v_R, v_quat);

    v_scale[0] += R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2];
    v_scale[1] += R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2];
    v_scale[2] += R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2];
}

template <typename T>
inline __device__ void quat_scale_to_preci_vjp(
    // fwd inputs
    const vec4<T> quat, const vec3<T> scale,
    // precompute
    const mat3<T> R,
    // grad outputs
    const mat3<T> v_preci,
    // grad inputs
    vec4<T> &v_quat, vec3<T> &v_scale) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    T sx = 1.0f / scale[0], sy = 1.0f / scale[1], sz = 1.0f / scale[2];

    // M = R * S
    mat3<T> S = mat3<T>(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3<T> M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3<T> v_M = (v_preci + glm::transpose(v_preci)) * M;
    mat3<T> v_R = v_M * S;

    // grad for (quat, scale) from preci
    quat_to_rotmat_vjp<T>(quat, v_R, v_quat);

    v_scale[0] +=
        -sx * sx * (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]);
    v_scale[1] +=
        -sy * sy * (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]);
    v_scale[2] +=
        -sz * sz * (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2]);
}

template <typename T>
inline __device__ void persp_proj(
    // inputs
    const vec3<T> mean3d, const mat3<T> cov3d, const T fx, const T fy, const T cx,
    const T cy, const uint32_t width, const uint32_t height,
    // outputs
    mat2<T> &cov2d, vec2<T> &mean2d, vec2<T> &ray_plane, vec3<T> &normal) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x = 1.3f * tan_fovx;
    T lim_y = 1.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T u = min(lim_x, max(-lim_x, x * rz));
    T v = min(lim_y, max(-lim_y, y * rz));
    T tx = z * u;
    T ty = z * v;

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(fx * rz, 0.f,                  // 1st column
                            0.f, fy * rz,                  // 2nd column
                            -fx * tx * rz2, -fy * ty * rz2 // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2<T>({fx * x * rz + cx, fy * y * rz + cy});

    // calculate the ray space intersection plane.
    auto length = [](T x, T y, T z) { return sqrt(x*x+y*y+z*z); };
    mat3<T> cov3d_eigen_vector;
	vec3<T> cov3d_eigen_value;
	int D = glm_modification::findEigenvaluesSymReal(cov3d,cov3d_eigen_value,cov3d_eigen_vector);
    unsigned int min_id = cov3d_eigen_value[0]>cov3d_eigen_value[1]? (cov3d_eigen_value[1]>cov3d_eigen_value[2]?2:1):(cov3d_eigen_value[0]>cov3d_eigen_value[2]?2:0);
    mat3<T> cov3d_inv;
	bool well_conditioned = cov3d_eigen_value[min_id]>1E-8;
	vec3<T> eigenvector_min;
	if(well_conditioned)
	{
		mat3<T> diag = mat3<T>( 1/cov3d_eigen_value[0], 0, 0,
                                0, 1/cov3d_eigen_value[1], 0,
                                0, 0, 1/cov3d_eigen_value[2] );
		cov3d_inv = cov3d_eigen_vector * diag * glm::transpose(cov3d_eigen_vector);
	}
	else
	{
		eigenvector_min = cov3d_eigen_vector[min_id];
		cov3d_inv = glm::outerProduct(eigenvector_min,eigenvector_min);
	}
    vec3<T> uvh = {u, v, 1};
    vec3<T> Cinv_uvh = cov3d_inv * uvh;
    if(length(Cinv_uvh.x, Cinv_uvh.y, Cinv_uvh.z) < 1E-12 || D ==0)
    {
        normal = {0, 0, 0};
        ray_plane = {0, 0};
    }
    else
    {
        T l = length(tx, ty, z);
        mat3<T> nJ_T = glm::mat3(rz, 0.f, -tx * rz2,             // 1st column
                                0.f, rz, -ty * rz2,              // 2nd column
                                tx/l, ty/l, z/l                  // 3rd column
        );
        T uu = u * u;
        T vv = v * v;
        T uv = u * v;

        mat3x2<T> nJ_inv_T = mat3x2<T>(vv + 1, -uv,          // 1st column
                                       -uv, uu + 1,          // 2nd column
                                        -u, -v               // 3nd column
        );
        T factor = l / (uu + vv + 1);
        vec3<T> Cinv_uvh_n = glm::normalize(Cinv_uvh);
        T u_Cinv_u_n_clmap = max(glm::dot(Cinv_uvh_n, uvh), 1E-7);
        vec2<T> plane = nJ_inv_T * (Cinv_uvh_n / u_Cinv_u_n_clmap);
        vec3<T> ray_normal_vector = {-plane.x*factor, -plane.y*factor, -1};
		vec3<T> cam_normal_vector = nJ_T * ray_normal_vector;
		normal = glm::normalize(cam_normal_vector);
        ray_plane = {plane.x * factor / fx, plane.y * factor / fy};
    }

}

template <typename T>
inline __device__ void persp_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d, const mat3<T> cov3d, const T fx, const T fy, const T cx,
    const T cy, const uint32_t width, const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d, const vec2<T> v_mean2d, const vec2<T> v_ray_plane, const vec3<T> v_normal,
    // grad inputs
    vec3<T> &v_mean3d, mat3<T> &v_cov3d) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x = 1.3f * tan_fovx;
    T lim_y = 1.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T u = min(lim_x, max(-lim_x, x * rz));
    T v = min(lim_y, max(-lim_y, y * rz));
    T tx = z * u;
    T ty = z * v;
    mat3<T> v_cov3d_ = {0,0,0,0,0,0,0,0,0};

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(fx * rz, 0.f,                  // 1st column
                            0.f, fy * rz,                  // 2nd column
                            -fx * tx * rz2, -fy * ty * rz2 // 3rd column
    );

    // calculate the ray space intersection plane.
    auto length = [](T x, T y, T z) { return sqrt(x*x+y*y+z*z); };
    mat3<T> cov3d_eigen_vector;
	vec3<T> cov3d_eigen_value;
	int D = glm_modification::findEigenvaluesSymReal(cov3d,cov3d_eigen_value,cov3d_eigen_vector);
    unsigned int min_id = cov3d_eigen_value[0]>cov3d_eigen_value[1]? (cov3d_eigen_value[1]>cov3d_eigen_value[2]?2:1):(cov3d_eigen_value[0]>cov3d_eigen_value[2]?2:0);
    mat3<T> cov3d_inv;
	bool well_conditioned = cov3d_eigen_value[min_id]>1E-8;
	vec3<T> eigenvector_min;
	if(well_conditioned)
	{
		mat3<T> diag = mat3<T>( 1/cov3d_eigen_value[0], 0, 0,
                                0, 1/cov3d_eigen_value[1], 0,
                                0, 0, 1/cov3d_eigen_value[2] );
		cov3d_inv = cov3d_eigen_vector * diag * glm::transpose(cov3d_eigen_vector);
	}
	else
	{
		eigenvector_min = cov3d_eigen_vector[min_id];
		cov3d_inv = glm::outerProduct(eigenvector_min,eigenvector_min);
	}
    vec3<T> uvh = {u, v, 1};
    vec3<T> Cinv_uvh = cov3d_inv * uvh;
    T l, v_u, v_v, v_l;
    mat3<T> v_nJ_T;
    if(length(Cinv_uvh.x, Cinv_uvh.y, Cinv_uvh.z) < 1E-12 || D ==0)
    {
        l = 1.f;
        v_u = 0.f;
        v_v = 0.f;
        v_l = 0.f;
        v_nJ_T = {0,0,0,0,0,0,0,0,0};
    }
    else
    {
        l = length(tx, ty, z);
        mat3<T> nJ_T = glm::mat3(rz, 0.f, -tx * rz2,             // 1st column
                                0.f, rz, -ty * rz2,              // 2nd column
                                tx/l, ty/l, z/l                  // 3rd column
        );
        T uu = u * u;
        T vv = v * v;
        T uv = u * v;

        mat3x2<T> nJ_inv_T = mat3x2<T>(vv + 1, -uv,          // 1st column
                                       -uv, uu + 1,          // 2nd column
                                        -u, -v               // 3nd column
        );
        const T nl = uu + vv + 1;
        T factor = l / nl;
        vec3<T> Cinv_uvh_n = glm::normalize(Cinv_uvh);
        T u_Cinv_u = glm::dot(Cinv_uvh, uvh);
        T u_Cinv_u_n = glm::dot(Cinv_uvh_n, uvh);
        T u_Cinv_u_clmap = max(u_Cinv_u, 1E-7);
        T u_Cinv_u_n_clmap = max(u_Cinv_u_n, 1E-7);
        mat3<T> cov3d_inv_u_Cinv_u = cov3d_inv / u_Cinv_u_clmap;
        vec3<T> Cinv_uvh_u_Cinv_u = Cinv_uvh_n / u_Cinv_u_n_clmap;
        vec2<T> plane = nJ_inv_T * Cinv_uvh_u_Cinv_u;
        vec3<T> ray_normal_vector = {-plane.x*factor, -plane.y*factor, -1};
		vec3<T> cam_normal_vector = nJ_T * ray_normal_vector;
		vec3<T> normal = glm::normalize(cam_normal_vector);
        vec2<T> ray_plane = {plane.x * factor / fx, plane.y * factor / fy};

        T cam_normal_vector_length = glm::length(cam_normal_vector);

        vec3<T> v_normal_l = v_normal / cam_normal_vector_length;
        vec3<T> v_cam_normal_vector = v_normal_l - normal * glm::dot(normal,v_normal_l);
        vec3<T> v_ray_normal_vector = glm::transpose(nJ_T) * v_cam_normal_vector;
        v_nJ_T = glm::outerProduct(v_cam_normal_vector, ray_normal_vector);

        // ray_plane_uv = {plane.x * factor, plane.y * factor};
        const vec2<T> v_ray_plane_uv = {v_ray_plane.x / fx, v_ray_plane.y / fy};
        v_l = glm::dot(plane, -glm::make_vec2(v_ray_normal_vector) + v_ray_plane_uv) / nl;
        vec2<T> v_plane = {factor * (-v_ray_normal_vector.x + v_ray_plane_uv.x),
                            factor * (-v_ray_normal_vector.y + v_ray_plane_uv.y)};
        T v_nl = (-v_ray_normal_vector.x * ray_normal_vector.x - v_ray_normal_vector.y * ray_normal_vector.y
                    -v_ray_plane.x*ray_plane.x - v_ray_plane.y*ray_plane.y) / nl;
        
        T tmp = glm::dot(v_plane, plane);
        if(well_conditioned)
        {
            v_cov3d_ += -glm::outerProduct(Cinv_uvh,
                            cov3d_inv_u_Cinv_u * (uvh * (-tmp) + glm::transpose(nJ_inv_T) * v_plane));
        }
        else
        {
            mat3<T> v_cov3d_inv = glm::outerProduct(uvh,
                                    (-tmp * uvh + glm::transpose(nJ_inv_T) * v_plane) / u_Cinv_u_clmap);
            vec3<T> v_eigenvector_min = (v_cov3d_inv + glm::transpose(v_cov3d_inv)) * eigenvector_min;
            for(int j =0;j<3;j++)
			{
				if(j!=min_id)
				{
					T scale = glm::dot(cov3d_eigen_vector[j], v_eigenvector_min)/min(cov3d_eigen_value[min_id] - cov3d_eigen_value[j], - 0.0000001f);
					v_cov3d_ += glm::outerProduct(cov3d_eigen_vector[j] * scale, eigenvector_min);
				}
			}
        }

        vec3<T> v_uvh = cov3d_inv_u_Cinv_u * (2 * (-tmp) * uvh +  glm::transpose(nJ_inv_T) * v_plane);
        mat3x2<T> v_nJ_inv_T = glm::outerProduct(v_plane, Cinv_uvh_u_Cinv_u);

        // derivative of u v in factor, uvh and nJ_inv_T variables.
        v_u = v_nl * 2 * u //dnl/du
            + v_uvh.x
            + (v_nJ_inv_T[0][1] + v_nJ_inv_T[1][0]) * (-v) + 2 * v_nJ_inv_T[1][1] * u - v_nJ_inv_T[2][0];
        v_v = v_nl * 2 * v //dnl/du
            + v_uvh.y
            + (v_nJ_inv_T[0][1] + v_nJ_inv_T[1][0]) * (-u) + 2 * v_nJ_inv_T[0][0] * v - v_nJ_inv_T[2][1];

    }

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d_ += glm::transpose(J) * v_cov2d * J;

    v_cov3d += v_cov3d_;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    v_mean3d += vec3<T>(fx * rz * v_mean2d[0], fy * rz * v_mean2d[1],
                        -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2);

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    T rz3 = rz2 * rz;
    mat3x2<T> v_J =
        v_cov2d * J * glm::transpose(cov3d) + glm::transpose(v_cov2d) * J * cov3d;

    // fov clipping
    T l3 = l * l * l;
    T v_mean3d_x = -fx * rz2 * v_J[2][0] + v_u * rz
                    - v_nJ_T[0][2]*rz2 + v_nJ_T[2][0]*(1/l-tx*tx/l3) + (v_nJ_T[2][1] * tx + v_nJ_T[2][2] * z)*(-tx/l3)
                    + v_l * tx / l;
    T v_mean3d_y = -fy * rz2 * v_J[2][1]  + v_v * rz
                    - v_nJ_T[1][2]*rz2 + (v_nJ_T[2][0]* tx + v_nJ_T[2][2]* z) *(-ty/l3) + v_nJ_T[2][1]*(1/l-ty*ty/l3)
                    + v_l * ty / l;
    if (x * rz <= lim_x && x * rz >= -lim_x) {
        v_mean3d.x += v_mean3d_x;
    } else {
        // v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
        v_mean3d.z += v_mean3d_x * u;
    }
    if (y * rz <= lim_y && y * rz >= -lim_y) {
        v_mean3d.y += v_mean3d_y;
    } else {
        // v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
        v_mean3d.z += v_mean3d_y * v;
    }
    v_mean3d.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1]
                  + 2.f * fx * tx * rz3 * v_J[2][0] + 2.f * fy * ty * rz3 * v_J[2][1]
                  - (v_u * tx + v_v * ty) * rz2
			      + v_nJ_T[0][0] * (-rz2) + v_nJ_T[1][1] * (-rz2) + v_nJ_T[0][2] * (2 * tx * rz3) + v_nJ_T[1][2] * (2 * ty * rz3)
				  + (v_nJ_T[2][0] * tx + v_nJ_T[2][1] * ty) * (-z/l3) + v_nJ_T[2][2] * (1 / l - z * z / l3)
				  + v_l * z / l;
}

template <typename T>
inline __device__ void pos_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const mat3<T> R, const vec3<T> t, const vec3<T> p, vec3<T> &p_c) {
    p_c = R * p + t;
}

template <typename T>
inline __device__ void pos_world_to_cam_vjp(
    // fwd inputs
    const mat3<T> R, const vec3<T> t, const vec3<T> p,
    // grad outputs
    const vec3<T> v_p_c,
    // grad inputs
    mat3<T> &v_R, vec3<T> &v_t, vec3<T> &v_p) {
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    v_R += glm::outerProduct(v_p_c, p);
    v_t += v_p_c;
    v_p += glm::transpose(R) * v_p_c;
}

template <typename T>
inline __device__ void covar_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const mat3<T> R, const mat3<T> covar, mat3<T> &covar_c) {
    covar_c = R * covar * glm::transpose(R);
}

template <typename T>
inline __device__ void covar_world_to_cam_vjp(
    // fwd inputs
    const mat3<T> R, const mat3<T> covar,
    // grad outputs
    const mat3<T> v_covar_c,
    // grad inputs
    mat3<T> &v_R, mat3<T> &v_covar) {
    // for D = W * X * WT, G = df/dD
    // df/dX = WT * G * W
    // df/dW
    // = G * (X * WT)T + ((W * X)T * G)T
    // = G * W * XT + (XT * WT * G)T
    // = G * W * XT + GT * W * X
    v_R +=
        v_covar_c * R * glm::transpose(covar) + glm::transpose(v_covar_c) * R * covar;
    v_covar += glm::transpose(R) * v_covar_c * R;
}

template <typename T> inline __device__ T inverse(const mat2<T> M, mat2<T> &Minv) {
    T det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (det <= 0.f) {
        return det;
    }
    T invDet = 1.f / det;
    Minv[0][0] = M[1][1] * invDet;
    Minv[0][1] = -M[0][1] * invDet;
    Minv[1][0] = Minv[0][1];
    Minv[1][1] = M[0][0] * invDet;
    return det;
}

template <typename T>
inline __device__ void inverse_vjp(const T Minv, const T v_Minv, T &v_M) {
    // P = M^-1
    // df/dM = -P * df/dP * P
    v_M += -Minv * v_Minv * Minv;
}

template <typename T>
inline __device__ T add_blur(const T eps2d, mat2<T> &covar, T &compensation) {
    T det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    T det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

template <typename T>
inline __device__ void add_blur_vjp(const T eps2d, const mat2<T> conic_blur,
                                    const T compensation, const T v_compensation,
                                    mat2<T> &v_covar) {
    // comp = sqrt(det(covar) / det(covar_blur))

    // d [det(M)] / d M = adj(M)
    // d [det(M + aI)] / d M  = adj(M + aI) = adj(M) + a * I
    // d [det(M) / det(M + aI)] / d M
    // = (det(M + aI) * adj(M) - det(M) * adj(M + aI)) / (det(M + aI))^2
    // = adj(M) / det(M + aI) - adj(M + aI) / det(M + aI) * comp^2
    // = (adj(M) - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M + aI) = adj(M) + a * I
    // = (adj(M + aI) - aI - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M) / det(M) = inv(M)
    // = (1 - comp^2) * inv(M + aI) - aI / det(M + aI)
    // given det(inv(M)) = 1 / det(M)
    // = (1 - comp^2) * inv(M + aI) - aI * det(inv(M + aI))
    // = (1 - comp^2) * conic_blur - aI * det(conic_blur)

    T det_conic_blur =
        conic_blur[0][0] * conic_blur[1][1] - conic_blur[0][1] * conic_blur[1][0];
    T v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    T one_minus_sqr_comp = 1 - compensation * compensation;
    v_covar[0][0] +=
        v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][0] - eps2d * det_conic_blur);
    v_covar[0][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][1]);
    v_covar[1][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][0]);
    v_covar[1][1] +=
        v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][1] - eps2d * det_conic_blur);
}

#endif // GSPLAT_CUDA_UTILS_H