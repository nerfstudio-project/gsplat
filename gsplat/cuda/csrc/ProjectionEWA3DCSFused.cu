#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

#define MAX_NB_POINTS 8
namespace gsplat {

// Helper function to compute cross product of two vectors OA and OB
// A positive cross product indicates a counterclockwise turn,
// a negative cross product indicates a clockwise turn,
// and a zero cross product indicates the points are collinear.
__forceinline__ __device__ float crossProduct(const float* O, const float* A, const float* B)
{
  return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
}
 
namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void projection_ewa_3dcs_fused_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ convex_points,              // [N, 3]
    const int32_t *__restrict__ cumsum_of_points_per_convex, // [N]
    const scalar_t *__restrict__ delta,                      // [N]
    const scalar_t *__restrict__ sigma,                      // [N]
    scalar_t *__restrict__ scaling, // [N] FIXME move it to outputs
    const scalar_t *__restrict__ opacities, // [N] optional
    const scalar_t *__restrict__ viewmats, // [C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ radii,         // [C, N, 2]
    scalar_t *__restrict__ normals,     // [C, total_nb_points, 2]
    scalar_t *__restrict__ offsets,     // [C, total_nb_points]
    scalar_t *__restrict__ p_image,      // [C, total_nb_points, 2]
    int32_t *__restrict__ hull,       // [C, 2*total_nb_points]
    int32_t *__restrict__ num_points_per_convex_view, // [C, N]
    int32_t *__restrict__ indices, // [C, total_nb_points]
    scalar_t *__restrict__ means2d,      // [C, N, 2]
    scalar_t *__restrict__ depths,       // [C, N]
    scalar_t *__restrict__ conics,       // [C, N, 3]
    scalar_t *__restrict__ compensations // [C, N] optional
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t conv_id = idx % N; // gaussian id

    //const int cumsum_for_convex = cumsum_of_points_per_convex[conv_id];

    // shift pointers to the current camera and 3D convex
    convex_points +=  conv_id * 6 * 3;//cumsum_for_convex * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    // FIXME: to add?
    radii[idx * 2] = 0;
    radii[idx * 2 + 1] = 0;
    num_points_per_convex_view[idx] = 0;

    // glm is column-major but input is row-major
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 translation = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // Compute 3D convex center projection (camera and screenspace).
    vec3 center_convex{0.0f, 0.0f, 0.0f};

    // FIXME: If the number of points per convex is expected to change, change this.
    int num_points_per_convex = 6;
    for (int i = 0; i < num_points_per_convex; i++)
    {
      indices[idx*6 + i] = i;
      center_convex.x += convex_points[3 * i];
      center_convex.y += convex_points[3 * i + 1];
      center_convex.z += convex_points[3 * i + 2];
    }

    center_convex.x /= num_points_per_convex;
    center_convex.y /= num_points_per_convex;
    center_convex.z /= num_points_per_convex;

    // transform 3D Convex center to camera space
    vec3 center_convex_c;
    posW2C(R, translation, center_convex, center_convex_c);
    if (center_convex_c.z < near_plane || center_convex_c.z > far_plane)
    {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // perspective projection
    vec2 center_convex_2D;
    mat2 covar2d;

    // FIXME: Not used?
    mat3 covar_c(0.f);

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj(
            center_convex_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            center_convex_2D
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj(
            center_convex_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            center_convex_2D
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj(
            center_convex_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            center_convex_2D
        );
        break;
    }

    // Now project every points from the 3D convex
    // TODO: Should be num_points_per_convex[idx]. See if it changes somewhere.
    // TODO: Could be merge with previous loop?

    // Calculation of points in 2D image space. We need to compute the cov3D.
    // float cov3D[6] = {0.0f};
    vec2 point_convex_2D[6];

    for (int i = 0; i < num_points_per_convex; i++)
    {
        vec3 convex_point(convex_points[3 * i], convex_points[3 * i + 1], convex_points[3 * i + 2]);
        vec3 convex_point_c{0.0f, 0.0f, 0.0f};
        posW2C(R, translation, convex_point, convex_point_c);

        // Now project in 2D and keep it.
        mat2 point_covar2d;

        switch (camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj(
                convex_point_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                point_covar2d,
                point_convex_2D[i]
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj(
                convex_point_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                point_covar2d,
                point_convex_2D[i]
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj(
                convex_point_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                point_covar2d,
                point_convex_2D[i]
            );
            break;
      }
      // Now add the projection.
      p_image[2*idx*6 + 2*i] = point_convex_2D[i].x;
      p_image[2*idx*6 + 2*i + 1] = point_convex_2D[i].y;
    }

    float max_distance = sqrtf(
        (p_image[2*idx*6] - center_convex_2D.x) * (p_image[2*idx*6] - center_convex_2D.x) +
        (p_image[2*idx*6 + 1] - center_convex_2D.y) * (p_image[2*idx*6 + 1] - center_convex_2D.y)
    );
    float2 ref_point{p_image[2*idx*6], p_image[2*idx*6 + 1]};

    // Find the furthest point from the center of the convex
    for (int i = 1; i < num_points_per_convex; i++)
    {
      float distance = sqrtf(
          (p_image[2*idx*6 + 2*i] - center_convex_2D.x) * (p_image[2*idx*6 + 2*i] - center_convex_2D.x) +
          (p_image[2*idx*6 + 2*i + 1] - center_convex_2D.y) * (p_image[2*idx*6 + 2*i + 1] - center_convex_2D.y)
      );

      // Update max_distance if the current distance is greater
      if (distance > max_distance)
      {
        max_distance = distance;
      }

      if (p_image[2*idx*6 + 2*i + 1] < ref_point.y || (p_image[2*idx*6 + 2*i + 1] == ref_point.y && p_image[2*idx*6 + 2*i] < ref_point.x))
      {
        ref_point = make_float2(p_image[2*idx*6 + 2*i], p_image[2*idx*6 + 2*i + 1]);
      }
    }

    if (max_distance > 10000.0f)
    {
      return;
    }

    // Sort the points based on their polar angle with respect to ref_point.
    // There exist definitely better sorting algos
    for (int i = 0; i < num_points_per_convex - 1; i++)
    {
      for (int j = i + 1; j < num_points_per_convex; j++)
      {
      	float angle1 = atan2f(p_image[2*idx*6 + 2*i + 1] - ref_point.y, p_image[2*idx*6 + 2*i] - ref_point.x);
      	float angle2 = atan2f(p_image[2*idx*6 + 2*j + 1] - ref_point.y, p_image[2*idx*6 + 2*j] - ref_point.x);
      	if (angle1 > angle2)
        {
      	  float2 temp{p_image[2*idx*6 + 2*i], p_image[2*idx*6 + 2*i + 1]};
      	  p_image[2*idx*6 + 2*i] = p_image[2*idx*6 + 2*j];
      	  p_image[2*idx*6 + 2*i + 1] = p_image[2*idx*6 + 2*j + 1];
      	  p_image[2*idx*6 + 2*j] = temp.x;
      	  p_image[2*idx*6 + 2*j + 1] = temp.y;

      	  // Swap their corresponding indices
      	  int temp_idx = indices[idx*6 + i];
      	  indices[idx*6 + i] = indices[idx*6 + j];
      	  indices[idx*6 + j] = temp_idx;
      	}
      }
    }

    // Now we apply the Graham scan algorithm to find the convex hull.
    int hull_size = 0;

    // Lower hull
    for (int i = 0; i < num_points_per_convex; i++)
    {
      while (hull_size >= 2 && crossProduct(&(p_image[2*idx*6 + 2*hull[2*idx*6 + hull_size - 2]]), &(p_image[2*idx*6 + 2*hull[2*idx*6 + hull_size - 1]]), &(p_image[2*idx*6 + 2*i])) <= 0)
      	hull_size--;
      hull[2*idx*6 + hull_size] = i;
      hull_size++;
    }

    //Upper hull
    int t = hull_size + 1;
    for (int i = num_points_per_convex - 2; i >= 0; i--)
    {
    	while (hull_size >= t && crossProduct(&(p_image[2*idx*6 + 2*hull[2*idx*6 + hull_size - 2]]), &(p_image[2*idx*6 + 2*hull[2*idx*6 + hull_size - 1]]), &(p_image[2*idx*6 + 2*i])) <= 0)
          hull_size--;
      hull[2*idx*6 + hull_size] = i;
      hull_size++;
    }

    float max_distance_off = 0.0f;
    float previous_offset = 0.0f;
    int counter = 0;
    float max_distance_x = (2.1f / (center_convex_c.z * center_convex_c.z * delta[conv_id] * sigma[conv_id]));

    for (int i = 0; i < hull_size - 1; i++)
    {
      // Points forming the segment
      float2 p1_conv{p_image[2*idx*6 + 2*hull[2*idx*6 + i]], p_image[2*idx*6 + 2*hull[2*idx*6 + i] + 1]};
      float2 p2_conv{p_image[2*idx*6 + 2*hull[2*idx*6 + (i + 1) % hull_size]], p_image[2*idx*6 + 2*hull[2*idx*6 + (i + 1) % hull_size] + 1]};

      // Calculate the normal vector (90-degree counterclockwise rotation)
      float2 normal = { p2_conv.y - p1_conv.y, -(p2_conv.x - p1_conv.x)};

      // Calculate the offset (dot product of normal vector and point p1)
      float offset = - (normal.x * p1_conv.x + normal.y * p1_conv.y);

      normals[2*idx*6 + 2*i] = normal.x;
      normals[2*idx*6 + 2*i + 1] = normal.y;
      offsets[idx*6 + i] = offset;

      if (normal.x * center_convex_2D.x + normal.y * center_convex_2D.y + offset < 0)
      {
        offset -= max_distance_x;
      }
      else
      {
        offset += max_distance_x;
      }

      if (i != 0)
      {
        float denominator = normal.x * normals[2*idx*6 + 2*(i-1) + 1] - normal.y * normals[2*idx*6 + 2*(i-1)]; // to avoid division by small numbers

        //  calculate the point of intersection between normals[i] and normals[i-1]
        float2 intersection_point = { (-offset * normals[2*idx*6 + 2*(i-1) + 1] + previous_offset * normal.y) / denominator, (-previous_offset * normal.x + offset * normals[2*idx*6 + 2*(i-1)]) / denominator};

        float angle = acosf( (normal.x * normals[2*idx*6 + 2*(i-1)] + normal.y * normals[2*idx*6 + 2*(i-1) + 1]) / (sqrtf(normal.x * normal.x + normal.y * normal.y) * sqrtf(normals[2*idx*6 + 2*(i-1)] * normals[2*idx*6 + 2*(i-1)] + normals[2*idx*6 + 2*(i-1) + 1] * normals[2*idx*6 + 2*(i-1) + 1])));

        float distance = sqrtf((intersection_point.x - center_convex_2D.x) * (intersection_point.x - center_convex_2D.x) + (intersection_point.y - center_convex_2D.y) * (intersection_point.y - center_convex_2D.y));

        if (angle > 0.1f && angle < 3.0f)
        {
          max_distance_off += distance;
          counter++;
        }
      }

      previous_offset = offset;
      num_points_per_convex_view[idx] = i + 1;
    }

    if (num_points_per_convex_view[idx] < 3 || counter == 0 || num_points_per_convex_view[idx] > num_points_per_convex)
    {
      radii[idx * 2] = 0;
      radii[idx * 2 + 1] = 0;
      num_points_per_convex_view[idx] = 0;
      return;
    }

    max_distance_off = max_distance_off / counter;
    max_distance = ceil(max(max_distance * 1.1f, max_distance_off));
  //   // END 3DCS


    // float extend = 3.33f;
    // if (opacities != nullptr) {
    //     float opacity = opacities[conv_id];
    //     // if (compensations != nullptr)
    //     // {
    //     //     // we assume compensation term will be applied later on.
    //     //     opacity *= compensation;
    //     // }
    //     if (opacity < ALPHA_THRESHOLD)
    //     {
    //         radii[idx * 2] = 0;
    //         radii[idx * 2 + 1] = 0;
    //         return;
    //     }
    //     // Compute opacity-aware bounding box.
    //     // https://arxiv.org/pdf/2402.00525 Section B.2
    //     extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    // }

    // // compute tight rectangular bounding box (non differentiable)
    // // https://arxiv.org/pdf/2402.00525
    // float radius_x = ceilf(extend * max_distance);
    // float radius_y = ceilf(extend * max_distance);

    if (max_distance <= radius_clip)
    // if (radius_x <= radius_clip && radius_y <= radius_clip)
    {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        num_points_per_convex_view[idx] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if (center_convex_2D.x + max_distance <= 0 || center_convex_2D.x - max_distance >= image_width ||
        center_convex_2D.y + max_distance <= 0 || center_convex_2D.y - max_distance >= image_height)
    // if (center_convex_2D.x + radius_x <= 0 || center_convex_2D.x - radius_x >= image_width ||
    //     center_convex_2D.y + radius_y <= 0 || center_convex_2D.y - radius_y >= image_height)
    {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        num_points_per_convex_view[idx] = 0;
        return;
    }

    // write to outputs
    radii[idx * 2] = (int32_t)max_distance;
    radii[idx * 2 + 1] = (int32_t)max_distance;
    means2d[idx * 2] = center_convex_2D.x;
    means2d[idx * 2 + 1] = center_convex_2D.y;
    depths[idx] = center_convex_c.z;
    conics[idx * 3] = 0;
    conics[idx * 3 + 1] = 0;
    conics[idx * 3 + 2] = 0;
}

void launch_projection_ewa_3dcs_fused_fwd_kernel(
    // inputs
    const at::Tensor convex_points,                // [N, 3]
    const at::Tensor cumsum_of_points_per_convex, // [N]
    const at::Tensor delta,                       // [N]
    const at::Tensor sigma,                       // [N]
    at::Tensor scaling,                     // [N]
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    at::Tensor radii,                       // [C, N, 2]
    at::Tensor normals,                     // [C, total_nb_points, 2]
    at::Tensor offsets,                     // [C, total_nb_points]
    at::Tensor p_image,                     // [C, total_nb_points, 2]
    at::Tensor hull,                        // [C, 2*total_nb_points]
    at::Tensor num_points_per_convex_view,  // [C, N]
    at::Tensor indices,                     // [C, total_nb_points]
    at::Tensor means2d,                     // [C, N, 2]
    at::Tensor depths,                      // [C, N]
    at::Tensor conics,                      // [C, N, 3]
    at::optional<at::Tensor> compensations  // [C, N] optional
) {
    uint32_t N = convex_points.size(0);    // number of 3D convexes
    uint32_t C = viewmats.size(0);         // number of cameras

    int64_t n_elements = C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        convex_points.scalar_type(),
        "projection_ewa_3dcs_fused_fwd_kernel",
        [&]() {
            projection_ewa_3dcs_fused_fwd_kernel<float>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    C,
                    N,
                    convex_points.data_ptr<float>(),
                    cumsum_of_points_per_convex.data_ptr<int32_t>(), // FIXME: Good type?
                    delta.data_ptr<float>(),
                    sigma.data_ptr<float>(),
                    scaling.data_ptr<float>(),
                    opacities.has_value() ? opacities.value().data_ptr<float>()
                                         : nullptr,
                    viewmats.data_ptr<float>(),
                    Ks.data_ptr<float>(),
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    camera_model,
                    radii.data_ptr<int32_t>(),
                    normals.data_ptr<float>(),
                    offsets.data_ptr<float>(),
                    p_image.data_ptr<float>(),
                    hull.data_ptr<int32_t>(),
                    num_points_per_convex_view.data_ptr<int32_t>(),
                    indices.data_ptr<int32_t>(),
                    means2d.data_ptr<float>(),
                    depths.data_ptr<float>(),
                    conics.data_ptr<float>(),
                    compensations.has_value()
                        ? compensations.value().data_ptr<float>()
                        : nullptr
                );
        }
    );
}

template <typename scalar_t>
__global__ void projection_ewa_3dcs_fused_bwd_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ convex_points,    // [N, K, 3]
    const int32_t *__restrict__ cumsum_of_points_per_convex, // [N]
    //	float* p_w,
    const scalar_t *__restrict__ viewmats,         // [C, 4, 4]
    const scalar_t *__restrict__ Ks,               // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const int32_t *__restrict__ radii,          // [C, N, 2]

    const int32_t *__restrict__ hull,       // [C, 2*total_nb_points]
    int32_t *__restrict__ num_points_per_convex_view, // [C, N]
    //const scalar_t *__restrict__ normals,     // [C, total_nb_points, 2]
    const scalar_t *__restrict__ offsets,     // [C, total_nb_points]
    const scalar_t *__restrict__ p_image,      // [C, total_nb_points, 2]
    const int32_t *__restrict__ indices, // [C, total_nb_points]

    const scalar_t *__restrict__ conics,        // [C, N, 3]
    const scalar_t *__restrict__ compensations, // [C, N] optional
    // grad outputs
    const scalar_t *__restrict__ v_means2d,       // [C, N, 2]
    const scalar_t *__restrict__ v_depths,        // [C, N]
    const scalar_t *__restrict__ v_conics,        // [C, N, 3]
    const scalar_t *__restrict__ v_compensations, // [C, N] optional
    // grad inputs
    scalar_t *__restrict__ v_convex_points,   // [N, K, 3]
    scalar_t *__restrict__ v_viewmats, // [C, 4, 4] optional
    const scalar_t *__restrict__ v_normals, // [C, total_nb_points, 2]
    const scalar_t *__restrict__ v_offsets // [C, total_nb_points]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t conv_id = idx % N; // convex id

    // shift pointers to the current camera and gaussian
    convex_points +=  conv_id * 6 * 3;//cumsum_for_convex * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;

    v_normals += idx * 6 * 2;
    v_offsets += idx * 6;

    // vjp: compute the inverse of the 2d covariance
    mat2 v_covar2d(0.f);

    // FIXME: If the number of points per convex is expected to change, change this.
    int num_points_per_convex = 6;

    // Initialize loss accumulators for normals and offsets
    float loss_points_x[MAX_NB_POINTS] = {0.0f};
    float loss_points_y[MAX_NB_POINTS] = {0.0f};

    // Iterate over the convex hull.
    for (int i = 0; i < num_points_per_convex_view[idx]; i++)
    {
      float v_normal_x = v_normals[2*i];
      float v_normal_y = v_normals[2*i + 1];
      float v_offset = v_offsets[i];

      float2 p1_conv{p_image[2*idx*6 + 2*hull[2*idx*6 + i]], p_image[2*idx*6 + 2*hull[2*idx*6 + i] + 1]};
      float2 p2_conv{p_image[2*idx*6 + 2*hull[2*idx*6 + (i + 1) % (num_points_per_convex_view[idx]+1)]], p_image[2*idx*6 + 2*hull[2*idx*6 + (i + 1) % (num_points_per_convex_view[idx]+1)] + 1]};

      // Calculate the normal vector (90-degree counterclockwise rotation)
      float2 normal = { p2_conv.y - p1_conv.y, -(p2_conv.x - p1_conv.x)};

      // Calculate the gradient of the loss with respect to the points p1 and p2
      // Gradient with respect to p1 (due to normal and offset)
      float2 v_p1_conv = { (v_normal_y - v_offset * p2_conv.y), (-v_normal_x + v_offset * p2_conv.x)};

      float2 v_p2_conv = {(-v_normal_y + v_offset * p1_conv.y), (v_normal_x - v_offset * p1_conv.x)};

      loss_points_x[indices[idx*6 + hull[2*idx*6 + i]]] += v_p1_conv.x;
      loss_points_y[indices[idx*6 + hull[2*idx*6 + i]]] += v_p1_conv.y;

      loss_points_x[indices[idx*6 + hull[2*idx*6 + (i + 1) % (num_points_per_convex_view[idx]+1)]]] += v_p2_conv.x;
      loss_points_y[indices[idx*6 + hull[2*idx*6 + (i + 1) % (num_points_per_convex_view[idx]+1)]]] += v_p2_conv.y;
    }

    vec2 v_means2d_tmp(0.f);
    v_means2d_tmp.x = 0;
    v_means2d_tmp.y = 0;

    // transform Gaussian to camera space
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 translation = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // vjp: perspective projection
    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];

    vec2 point_convex_2D[6];
    vec3 v_mean[6];
    for (int i = 0; i < num_points_per_convex; i++)
    {
        v_means2d_tmp.x += loss_points_x[i];
        v_means2d_tmp.y += loss_points_y[i];
        vec2 loss_point{loss_points_x[i], loss_points_y[i]};
        vec3 convex_point(convex_points[3 * i], convex_points[3 * i + 1], convex_points[3 * i + 2]);
        vec3 convex_point_c{0.0f, 0.0f, 0.0f};
        posW2C(R, translation, convex_point, convex_point_c);

        mat3 covar_c(0.f);

        mat3 v_covar_c(0.f);
        vec3 v_convex_point_c(0.f);

        switch (camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj_vjp(
                convex_point_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                v_covar2d,
                loss_point,
                v_convex_point_c,
                v_covar_c
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj_vjp(
                convex_point_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                v_covar2d,
                loss_point,
                v_convex_point_c,
                v_covar_c
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj_vjp(
                convex_point_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                v_covar2d,
                loss_point,
                v_convex_point_c,
                v_covar_c
            );
            break;
        }

        // add contribution from v_depths --> Basically put the correct Z to the point.
        v_convex_point_c.z += v_depths[0];

        // vjp: transform Gaussian covariance to camera space
        v_mean[i] = vec3(0.f);
        mat3 v_covar(0.f);
        mat3 v_R(0.f);
        vec3 v_t(0.f);
        posW2C_VJP(R, translation, convex_point, v_convex_point_c, v_R, v_t, v_mean[i]);
    }

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, conv_id);
    if (v_convex_points != nullptr)
    {
        // FIXME: Reactivate this!
        //warpSum<MAX_NB_POINTS, vec3>(v_mean, warp_group_g);
        warpSum<MAX_NB_POINTS>(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0)
        {
            float *v_convex_points_ptr = (float *)(v_convex_points) + 6*conv_id*3;
#pragma unroll
            for (int i = 0; i < num_points_per_convex; i++)
            {
#pragma unroll
                for (uint32_t h = 0; h < 3; h++)
                {
                    gpuAtomicAdd(v_convex_points_ptr + h + 3*i, v_mean[i][h]);
                }
            }
        }
    }

//     if (v_viewmats != nullptr) {
//         auto warp_group_c = cg::labeled_partition(warp, cid);
//         warpSum(v_R, warp_group_c);
//         warpSum(v_t, warp_group_c);
//         if (warp_group_c.thread_rank() == 0) {
//             v_viewmats += cid * 16;
// #pragma unroll
//             for (uint32_t i = 0; i < 3; i++) { // rows
// #pragma unroll
//                 for (uint32_t j = 0; j < 3; j++) { // cols
//                     gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
//                 }
//                 gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
//             }
//         }
//     }
}

void launch_projection_ewa_3dcs_fused_bwd_kernel(
    // inputs
    // fwd inputs
    const at::Tensor convex_points,                // [N, K, 3]
    const at::Tensor cumsum_of_points_per_convex, // [N]
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [C, N, 2]
    const at::Tensor hull,                        // [C, 2*total_nb_points]
    const at::Tensor num_points_per_convex_view,  // [C, N]
    //const at::Tensor normals,                     // [C, total_nb_points, 2]
    const at::Tensor offsets,                     // [C, total_nb_points]
    const at::Tensor p_image,                     // [C, total_nb_points, 2]
    const at::Tensor indices,                     // [C, total_nb_points]
    const at::Tensor conics,                      // [C, N, 3]
    const at::optional<at::Tensor> compensations, // [C, N] optional
    // grad inputs
    const at::Tensor v_normals,                     // [C, total_nb_points, 2]
    const at::Tensor v_offsets,                     // [C, total_nb_points]
    // grad outputs
    const at::Tensor v_means2d,                     // [C, N, 2]
    const at::Tensor v_depths,                      // [C, N]
    const at::Tensor v_conics,                      // [C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [C, N] optional
    const bool viewmats_requires_grad,
    // outputs
    at::Tensor v_convex_points,   // [N, K, 3]
    at::Tensor v_viewmats         // [C, 4, 4]
) {
    uint32_t N = convex_points.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras

    int64_t n_elements = C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // FIXME: Why does is it a double here?
    AT_DISPATCH_FLOATING_TYPES(
        convex_points.scalar_type(),
        "projection_ewa_3dcs_fused_bwd_kernel",
        [&]() {
            projection_ewa_3dcs_fused_bwd_kernel<float>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    C,
                    N,
                    convex_points.data_ptr<float>(),
                    cumsum_of_points_per_convex.data_ptr<int32_t>(), // FIXME: Good type?
                    viewmats.data_ptr<float>(),
                    Ks.data_ptr<float>(),
                    image_width,
                    image_height,
                    eps2d,
                    camera_model,
                    radii.data_ptr<int32_t>(),
                    hull.data_ptr<int32_t>(),
                    num_points_per_convex_view.data_ptr<int32_t>(),
                    // normals.data_ptr<float>(),
                    offsets.data_ptr<float>(),
                    p_image.data_ptr<float>(),
                    indices.data_ptr<int32_t>(),
                    conics.data_ptr<float>(),
                    compensations.has_value()
                        ? compensations.value().data_ptr<float>()
                        : nullptr,
                    v_means2d.data_ptr<float>(),
                    v_depths.data_ptr<float>(),
                    v_conics.data_ptr<float>(),
                    v_compensations.has_value()
                        ? v_compensations.value().data_ptr<float>()
                        : nullptr,
                    v_convex_points.data_ptr<float>(),
                    viewmats_requires_grad ? v_viewmats.data_ptr<float>()
                                           : nullptr,
                    v_normals.data_ptr<float>(),
                    v_offsets.data_ptr<float>()
                );
        }
    );
}

} // namespace gsplat
