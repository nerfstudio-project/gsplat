#ifndef GSPLAT_CUDA_TETRA_CUH
#define GSPLAT_CUDA_TETRA_CUH

#include "types.cuh"

namespace gsplat {

template <typename T>
inline __device__ int sign(T x) {
    return (x > 0) - (x < 0);
}

// Function to check ray-triangle intersection using Möller–Trumbore algorithm
template <typename T>
__device__ bool ray_triangle_intersection(
    // ray origin and direction
    const vec3<T> o,
    const vec3<T> d,
    // triangle vertices
    const vec3<T> v0,
    const vec3<T> v1,
    const vec3<T> v2,
    // output intersection t value
    T &t
) {
    const T EPSILON = 0.00000001f;
    t = 0.0f;
    // e1 = v1 - v0
    // e2 = v2 - v0
    vec3<T> e1 = v1 - v0;
    vec3<T> e2 = v2 - v0;

    // h = cross(d, e2)
    // a = dot(e1, h)
    vec3<T> h = glm::cross(d, e2);
    T a = glm::dot(e1, h);
    if (a > -EPSILON && a < EPSILON) return false; // parallel to the triangle

    // f = 1 / a
    // s = o - v0
    // u = f * dot(s, h)
    T f = 1.0f / a;
    vec3<T> s = o - v0;
    T dot_s_h = glm::dot(s, h);
    T u = f * dot_s_h;
    if (u < 0.0f || u > 1.0f) return false; // outside the triangle

    // q = cross(s, e1)
    // v = f * dot(d, q)
    vec3<T> q = glm::cross(s, e1);
    T dot_d_q = glm::dot(d, q);
    T v = f * dot_d_q;
    if (v < 0.0f || u + v > 1.0f) return false; // outside the triangle

    // t = f * dot(e2, q)
    T dot_e2_q = glm::dot(e2, q);
    t = f * dot_e2_q;

    return true;
    // if (t > EPSILON) return true;
    // return false; // There is a line intersection, but not a ray intersection
}

template <typename T>
__device__ void ray_triangle_intersection_vjp(
    // fwd inputs
    const vec3<T> o,
    const vec3<T> d,
    const vec3<T> v0,
    const vec3<T> v1,
    const vec3<T> v2,
    // grad outputs
    const T &v_t,
    // grad inputs (only backpropagate to triangle vertices)
    vec3<T> &v_v0,
    vec3<T> &v_v1,
    vec3<T> &v_v2
) {
    v_v0 = vec3<T>(0.f);
    v_v1 = vec3<T>(0.f);
    v_v2 = vec3<T>(0.f);

    // we call this function only when there is a ray-triangle intersection
    // so the forward is only used to compute intermediate variables
    vec3<T> e1 = v1 - v0;
    vec3<T> e2 = v2 - v0;
    vec3<T> h = glm::cross(d, e2);
    T a = glm::dot(e1, h);
    T f = 1.0f / a;
    vec3<T> s = o - v0;
    T dot_s_h = glm::dot(s, h);
    T u = f * dot_s_h;
    vec3<T> q = glm::cross(s, e1);
    T dot_d_q = glm::dot(d, q);
    T v = f * dot_d_q;
    T dot_e2_q = glm::dot(e2, q);
    T t = f * dot_e2_q;

    vec3<T> v_e1, v_e2, v_h, v_s, v_q;
    T v_a, v_f;

    // t = f * dot(e2, q), 
    v_f = v_t * dot_e2_q;
    v_e2 = v_t * q * f;
    v_q = v_t * e2 * f;
    
    // v = f * dot(d, q), v is leaf variable
    // q = cross(s, e1)
    v_s = glm::cross(e1, v_q);
    v_e1 = glm::cross(v_q, s);

    // u = f * dot(s, h), u is leaf variable
    // s = o - v0
    // v_o += v_s;
    v_v0 -= v_s;

    // f = 1 / a
    v_a = -v_f / (a * a);

    // a = dot(e1, h)
    v_e1 += v_a * h;
    v_h = v_a * e1;

    // h = cross(d, e2)
    // v_d += glm::cross(e2, v_h);
    v_e2 += glm::cross(v_h, d);

    // e2 = v2 - v0
    v_v2 += v_e2;
    v_v0 -= v_e2;
    
    // e1 = v1 - v0
    v_v1 += v_e1;
    v_v0 -= v_e1;
}

template <typename T>
__device__ bool ray_tetra_intersection(
    // ray origin and direction
    const vec3<T> o,
    const vec3<T> d,
    // tetrahedron vertices
    const vec3<T> v0,
    const vec3<T> v1,
    const vec3<T> v2,
    const vec3<T> v3,
    // output intersection face indices and t values
    int32_t &entry_face_idx, // entry face index
    int32_t &exit_face_idx, // exit face index
    T &t_entry, // entry face t value
    T &t_exit  // exit face t value
) {
    entry_face_idx = -1;
    exit_face_idx = -1;
    t_entry = 1e10f;
    t_exit = -1e10f;

    // Test intersection with each of the four faces
    T t_isct;
    vec3<T> faces[4][3] = {
        {v0, v1, v2}, // Face 0
        {v1, v2, v3}, // Face 1
        {v2, v3, v0}, // Face 2
        {v3, v0, v1}  // Face 3
    };

    GSPLAT_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
        bool intersected_face = ray_triangle_intersection(
            o,
            d,
            faces[i][0],
            faces[i][1],
            faces[i][2],
            t_isct
        );
        if (intersected_face) {
            if (t_isct < t_entry) {
                entry_face_idx = i;
                t_entry = t_isct;
            }
            if (t_isct > t_exit) {
                exit_face_idx = i;
                t_exit = t_isct;
            }
        }
    }
    if (entry_face_idx < 0 && exit_face_idx < 0) return false;
    return true;
}

template <typename T>
__device__ void ray_tetra_intersection_vjp(
    // fwd inputs
    const vec3<T> o,
    const vec3<T> d,
    const vec3<T> v0,
    const vec3<T> v1,
    const vec3<T> v2,
    const vec3<T> v3,
    // grad outputs
    const T &v_t_entry,
    const T &v_t_exit,
    // grad inputs (only backpropagate to tetrahedron vertices)
    vec3<T> &v_v0,
    vec3<T> &v_v1,
    vec3<T> &v_v2,
    vec3<T> &v_v3
) {
    if (v_t_entry == 0 && v_t_exit == 0) return;

    v_v0 = vec3<T>(0.f);
    v_v1 = vec3<T>(0.f);
    v_v2 = vec3<T>(0.f);
    v_v3 = vec3<T>(0.f);

    // run forward pass to get intersection face indices
    int32_t entry_face_idx = -1;
    int32_t exit_face_idx = -1;
    T t_entry = 1e10f;
    T t_exit = -1e10f;
    bool hit = ray_tetra_intersection(
        o,
        d,
        v0,
        v1,
        v2,
        v3,
        entry_face_idx,
        exit_face_idx,
        t_entry,
        t_exit
    );
    if (!hit) return;

    vec3<T> faces[4][3] = {
        {v0, v1, v2}, // Face 0
        {v1, v2, v3}, // Face 1
        {v2, v3, v0}, // Face 2
        {v3, v0, v1}  // Face 3
    };

    // backpropagate to the tetrahedron vertices

    vec3<T> v_entry_faces[3] = {vec3<T>(0.f), vec3<T>(0.f), vec3<T>(0.f)};
    ray_triangle_intersection_vjp(
        o,
        d,
        faces[entry_face_idx][0],
        faces[entry_face_idx][1],
        faces[entry_face_idx][2],
        v_t_entry,
        v_entry_faces[0],
        v_entry_faces[1],
        v_entry_faces[2]
    );
    switch (entry_face_idx) {
        case 0:
            v_v0 += v_entry_faces[0];
            v_v1 += v_entry_faces[1];
            v_v2 += v_entry_faces[2];
            break;
        case 1:
            v_v1 += v_entry_faces[0];
            v_v2 += v_entry_faces[1];
            v_v3 += v_entry_faces[2];
            break;
        case 2:
            v_v2 += v_entry_faces[0];
            v_v3 += v_entry_faces[1];
            v_v0 += v_entry_faces[2];
            break;
        case 3:
            v_v3 += v_entry_faces[0];
            v_v0 += v_entry_faces[1];
            v_v1 += v_entry_faces[2];
            break;
    }

    vec3<T> v_exit_faces[3] = {vec3<T>(0.f), vec3<T>(0.f), vec3<T>(0.f)};
    ray_triangle_intersection_vjp(
        o,
        d,
        faces[exit_face_idx][0],
        faces[exit_face_idx][1],
        faces[exit_face_idx][2],
        v_t_exit,
        v_exit_faces[0],
        v_exit_faces[1],
        v_exit_faces[2]
    );
    switch (exit_face_idx) {
        case 0:
            v_v0 += v_exit_faces[0];
            v_v1 += v_exit_faces[1];
            v_v2 += v_exit_faces[2];
            break;
        case 1:
            v_v1 += v_exit_faces[0];
            v_v2 += v_exit_faces[1];
            v_v3 += v_exit_faces[2];
            break;
        case 2:
            v_v2 += v_exit_faces[0];
            v_v3 += v_exit_faces[1];
            v_v0 += v_exit_faces[2];
            break;
        case 3:
            v_v3 += v_exit_faces[0];
            v_v0 += v_exit_faces[1];
            v_v1 += v_exit_faces[2];
            break;
    }
}

} // namespace gsplat

#endif // GSPLAT_CUDA_TETRA_CUH