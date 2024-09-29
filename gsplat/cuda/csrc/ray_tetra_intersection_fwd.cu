#include "bindings.h"
#include "helpers.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace gsplat {

namespace cg = cooperative_groups;

template <typename T>
__global__ void ray_tetra_intersection_fwd_kernel(
    const uint32_t N,
    const T *__restrict__ rays_o,  // [N, 3]
    const T *__restrict__ rays_d,  // [N, 3]
    const T *__restrict__ vertices,   // [N, 4, 3]
    // outputs
    int32_t *__restrict__ isect_faces, // [N]
    T *__restrict__ isect_bary1, // [N, 2]
    T *__restrict__ isect_bary2, // [N, 2]
    T *__restrict__ isect_t1, // [N]
    T *__restrict__ isect_t2  // [N]
) {
    // parallelize over N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    // shift pointers to the current ray and tetra
    rays_o += idx * 3;
    rays_d += idx * 3;
    vertices += idx * 12;

    isect_faces += idx;
    isect_bary1 += idx * 2;
    isect_bary2 += idx * 2;
    isect_t1 += idx;
    isect_t2 += idx;

    // calculate intersection
    vec2<int> faces(0);
    vec2<T> bary1(0.f);
    vec2<T> bary2(0.f);
    T t1, t2;
    bool hit = ray_tetra_intersection(
        glm::make_vec3(rays_o),
        glm::make_vec3(rays_d),
        glm::make_vec3(vertices),
        glm::make_vec3(vertices + 3),
        glm::make_vec3(vertices + 6),
        glm::make_vec3(vertices + 9),
        // outputs
        faces,
        bary1,
        bary2,
        t1,
        t2
    );

    // store results
    isect_faces[0] = faces.x;
    isect_faces[1] = faces.y;
    isect_bary1[0] = bary1.x;
    isect_bary1[1] = bary1.y;
    isect_bary2[0] = bary2.x;
    isect_bary2[1] = bary2.y;
    isect_t1[0] = t1;
    isect_t2[0] = t2;
}

std::vector<torch::Tensor> ray_tetra_intersection_fwd_tensor(
    const torch::Tensor &rays_o,  // [N, 3]
    const torch::Tensor &rays_d,  // [N, 3]
    const torch::Tensor &vertices  // [N, 4, 3]
) {
    GSPLAT_DEVICE_GUARD(rays_o);
    GSPLAT_CHECK_INPUT(rays_o);
    GSPLAT_CHECK_INPUT(rays_d);
    GSPLAT_CHECK_INPUT(vertices);

    uint32_t N = rays_o.size(0);

    torch::Tensor faces = torch::empty({N}, rays_o.options().dtype(torch::kInt32));
    torch::Tensor bary1 = torch::empty({N, 2}, rays_o.options());
    torch::Tensor bary2 = torch::empty({N, 2}, rays_o.options());
    torch::Tensor isect_t1 = torch::empty({N}, rays_o.options());
    torch::Tensor isect_t2 = torch::empty({N}, rays_o.options());

    if (N > 0) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        ray_tetra_intersection_fwd_kernel<float>
            <<<(N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                GSPLAT_N_THREADS,
                0,
                stream>>>(
                N,
                rays_o.data_ptr<float>(),
                rays_d.data_ptr<float>(),
                vertices.data_ptr<float>(),
                faces.data_ptr<int32_t>(),
                bary1.data_ptr<float>(),
                bary2.data_ptr<float>(),
                isect_t1.data_ptr<float>(),
                isect_t2.data_ptr<float>()
            );
    }
    return {faces, bary1, bary2, isect_t1, isect_t2};
}


} // namespace gsplat















// // Segura Algorithm with Scalar Triple Product Calc
// // Reference:
// // https://github.com/JKolios/RayTetra/blob/master/RayTetra/RayTetraSTP0.cl
// inline __host__ __device__ void _ray_tetra_intersect(
//     const float *ray_o,
//     const float *ray_d,
//     const float *vert0,
//     const float *vert1,
//     const float *vert2,
//     const float *vert3,
//     float *t_min,
//     float *t_max
//     // scalar_t* cartesian,
//     // scalar_t* barycentric,
//     // scalar_t* parametric
// ) {
//     float3 ori = make_float3(ray_o[0], ray_o[1], ray_o[2]);
//     float3 dir = make_float3(ray_d[0], ray_d[1], ray_d[2]); // dir = Q
//     float3 v0 = make_float3(vert0[0], vert0[1], vert0[2]);
//     float3 v1 = make_float3(vert1[0], vert1[1], vert1[2]);
//     float3 v2 = make_float3(vert2[0], vert2[1], vert2[2]);
//     float3 v3 = make_float3(vert3[0], vert3[1], vert3[2]);

//     // Initialization of output vars
//     int leaveface = -1;
//     int enterface = -1;
//     float uLeave1 = 0;
//     float uLeave2 = 0;
//     float uEnter1 = 0;
//     float uEnter2 = 0;
//     float tLeave = 0;
//     float tEnter = 0;

//     float3 leavePoint = make_float3(0.f, 0.f, 0.f);
//     float3 enterPoint = make_float3(0.f, 0.f, 0.f);

//     // Point Translation
//     float3 A = v0 - ori;
//     float3 B = v1 - ori;
//     float3 C = v2 - ori;
//     float3 D = v3 - ori;

//     // Initial cross product calculation
//     float3 QxA = cross(dir, A);
//     float3 QxB = cross(dir, B);

//     // Intersection test with face ABC(face3)
//     // STPs needed:[QAB][QBC][QCA]
//     //[QAB]
//     float QAB = dot(QxA, B);
//     int signQAB = __sign(QAB);

//     //[QBC]
//     float QBC = dot(QxB, C);
//     int signQBC = __sign(QBC);

//     //[QAC] = - [QCA]
//     float QAC = dot(QxA, C);
//     int signQAC = __sign(QAC);

//     if ((signQAB != 0) && (signQAC != 0) && (signQBC != 0)) {
//         float invVolABC = 1.0 / (QAB + QBC - QAC);
//         if ((signQAB > 0.0) && (signQAC < 0.0) && (signQBC > 0.0)) {
//             // face 3 is the entry face
//             leaveface = 3;
//             uLeave1 = -QAC * invVolABC;
//             uLeave2 = QAB * invVolABC;
//             leavePoint =
//                 (1 - uLeave1 - uLeave2) * v0 + uLeave1 * v1 + uLeave2 * v2;
//         } else if ((signQAB < 0.0) && (signQAC > 0.0) && (signQBC < 0.0)) {
//             // face 3 is the exit face
//             enterface = 3;
//             uEnter1 = -QAC * invVolABC;
//             uEnter2 = QAB * invVolABC;
//             enterPoint =
//                 (1 - uEnter1 - uEnter2) * v0 + uEnter1 * v1 + uEnter2 * v2;
//         }
//     }

//     // Intersection test with face BAD(face2)
//     // STPs needed:[QBA][QAD][QDB]
//     // Available: [QAB][QBC][QAC]

//     //[QBA] == - [QAB]

//     //[QAD]
//     float QAD = dot(QxA, D);
//     int signQAD = __sign(QAD);

//     //[QDB] == - [QBD]
//     //[QBD] is selected in order to avoid computing QxD.
//     float QBD = dot(QxB, D);
//     int signQBD = __sign(QBD);

//     if ((signQAB != 0) && (signQAD != 0) && (signQBD != 0)) {
//         float invVolBAD = 1.0 / (QAD - QBD - QAB);
//         if ((leaveface == -1) && (signQAB < 0.0) && (signQAD > 0.0) &&
//             (signQBD < 0.0)) {
//             // face 2 is the entry face
//             leaveface = 2;
//             uLeave1 = -QBD * invVolBAD;
//             uLeave2 = -QAB * invVolBAD;
//             leavePoint =
//                 (1 - uLeave1 - uLeave2) * v1 + uLeave1 * v0 + uLeave2 * v3;
//         } else if ((enterface == -1) && (signQAB > 0.0) && (signQAD < 0.0) &&
//                    (signQBD > 0.0)) {
//             // face 2 is the exit face
//             enterface = 2;
//             uEnter1 = -QBD * invVolBAD;
//             uEnter2 = -QAB * invVolBAD;
//             enterPoint =
//                 (1 - uEnter1 - uEnter2) * v1 + uEnter1 * v0 + uEnter2 * v3;
//         }
//     }

//     // Intersection test with face CDA(face1)
//     // STPs needed:[QCD][QDA][QAC]
//     // Available: [QAB][QBC][QAC][QAD][QBD]

//     //[QCD]
//     float3 QxC = cross(dir, C);
//     float QCD = dot(QxC, D);
//     int signQCD = __sign(QCD);

//     //[QDA] = - [QAD]

//     if ((signQAD != 0) && (signQAC != 0) && (signQCD != 0)) {
//         float invVolCDA = 1.0 / (QAC + QCD - QAD);
//         if ((leaveface == -1) && (signQAD < 0.0) && (signQAC > 0.0) &&
//             (signQCD > 0.0)) {
//             // face 1 is the entry face
//             leaveface = 1;
//             uLeave1 = QAC * invVolCDA;
//             uLeave2 = QCD * invVolCDA;
//             leavePoint =
//                 (1 - uLeave1 - uLeave2) * v2 + uLeave1 * v3 + uLeave2 * v0;
//         } else if ((enterface == -1) && (signQAD > 0.0) && (signQAC < 0.0) &&
//                    (signQCD < 0.0)) {
//             // face 1 is the exit face
//             enterface = 1;
//             uEnter1 = QAC * invVolCDA;
//             uEnter2 = QCD * invVolCDA;
//             enterPoint =
//                 (1 - uEnter1 - uEnter2) * v2 + uEnter1 * v3 + uEnter2 * v0;
//         }
//     }

//     // If no intersecting face has been found so far, then there is none,
//     // otherwise DCB (face0) is the remaining one
//     // STPs relevant to DCB:[QDC][QCB][QBD]
//     // Available: [QAB][QBC][QAC][QAD][QBD][QCD]

//     if ((signQBC != 0) && (signQBD != 0) && (signQCD != 0)) {
//         float invVolDCB = 1.0 / (-QCD - QBC + QBD);
//         if ((leaveface == -1) && (signQBC < 0.0) && (signQBD > 0.0) &&
//             (signQCD < 0.0)) {
//             // face 0 is the entry face
//             leaveface = 0;
//             uLeave1 = QBD * invVolDCB;
//             uLeave2 = -QCD * invVolDCB;
//             leavePoint =
//                 (1 - uLeave1 - uLeave2) * v3 + uLeave1 * v2 + uLeave2 * v1;
//         } else if ((enterface == -1) && (signQBC > 0.0) && (signQBD < 0.0) &&
//                    (signQCD > 0.0)) {
//             // face 0 is the exit face
//             enterface = 0;
//             uEnter1 = QBD * invVolDCB;
//             uEnter2 = -QCD * invVolDCB;
//             enterPoint =
//                 (1 - uEnter1 - uEnter2) * v3 + uEnter1 * v2 + uEnter2 * v1;
//         }
//     }

//     // Calculating the parametric distances tLeave and tEnter

//     if (dir.x) {
//         float invDirx = 1.0 / dir.x;
//         tLeave = (leavePoint.x - ori.x) * invDirx;
//         tEnter = (enterPoint.x - ori.x) * invDirx;
//     } else if (dir.y) {
//         float invDiry = 1.0 / dir.y;
//         tLeave = (leavePoint.y - ori.y) * invDiry;
//         tEnter = (enterPoint.y - ori.y) * invDiry;
//     } else {
//         float invDirz = 1.0 / dir.z;
//         tLeave = (leavePoint.z - ori.z) * invDirz;
//         tEnter = (enterPoint.z - ori.z) * invDirz;
//     }

//     tLeave = (leaveface != -1) ? tLeave : 0;
//     tEnter = (enterface != -1) ? tEnter : 0;

//     // cartesian[0] = enterface;
//     // cartesian[1] = leaveface;
//     // cartesian[2] = enterPoint.x;
//     // cartesian[3] = enterPoint.y;
//     // cartesian[4] = enterPoint.z;
//     // cartesian[5] = leavePoint.x;
//     // cartesian[6] = leavePoint.y;
//     // cartesian[7] = leavePoint.z;
//     // barycentric[0] = uEnter1;
//     // barycentric[1] = uEnter2;
//     // barycentric[2] = uLeave1;
//     // barycentric[3] = uLeave2;
//     // parametric[0] = tEnter;
//     // parametric[1] = tLeave;
//     *t_min = tEnter;
//     *t_max = tLeave;
// }

// __global__ void kernel_ray_tetra_intersect(
//     const int num_rays,
//     const int num_tetra,
//     const float *rays_o,
//     const float *rays_d,
//     const float *verts,
//     float *t_min,
//     float *t_max
// ) {
//     CUDA_GET_THREAD_ID(thread_id, num_rays * num_tetra);

//     const int ray_id = thread_id / num_tetra;
//     const int tetra_id = thread_id % num_tetra;

//     // locate
//     rays_o += ray_id * 3;
//     rays_d += ray_id * 3;
//     verts += tetra_id * (4 * 3);
//     t_min += thread_id;
//     t_max += thread_id;

//     _ray_tetra_intersect(
//         rays_o,
//         rays_d,
//         &verts[0 * 3],
//         &verts[1 * 3],
//         &verts[2 * 3],
//         &verts[3 * 3],
//         t_min,
//         t_max
//     );
//     return;
// }

// std::vector<torch::Tensor> ray_tetra_intersect(
//     const torch::Tensor rays_o,
//     const torch::Tensor rays_d,
//     const torch::Tensor verts
// ) {
//     DEVICE_GUARD(rays_o);
//     TORCH_CHECK(rays_o.ndimension() == 2 & rays_o.size(1) == 3)
//     TORCH_CHECK(rays_d.ndimension() == 2 & rays_d.size(1) == 3)
//     TORCH_CHECK(
//         verts.ndimension() == 3 & verts.size(1) == 4 && verts.size(2) == 3
//     )

//     const int num_rays = rays_o.size(0);
//     const int num_tetra = verts.size(0);

//     const int threads = 256;
//     const int blocks = CUDA_N_BLOCKS_NEEDED(num_rays * num_tetra, threads);

//     torch::Tensor t_min = torch::empty({num_rays, num_tetra}, rays_o.options());
//     torch::Tensor t_max = torch::empty({num_rays, num_tetra}, rays_o.options());

//     kernel_ray_tetra_intersect<<<blocks, threads>>>(
//         num_rays,
//         num_tetra,
//         rays_o.data_ptr<float>(),
//         rays_d.data_ptr<float>(),
//         verts.data_ptr<float>(),
//         t_min.data_ptr<float>(),
//         t_max.data_ptr<float>()
//     );

//     return {t_min, t_max};
// }