#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Cameras.cuh"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_from_world_3dgs_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec3 *__restrict__ means,       // [N, 3]
    const vec4 *__restrict__ quats,       // [N, 4]
    const vec3 *__restrict__ scales,      // [N, 3]
    const scalar_t *__restrict__ colors,      // [C, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [C, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [C, CDIM]
    const bool *__restrict__ masks,           // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // camera model
    const scalar_t *__restrict__ viewmats0, // [C, 4, 4]
    const scalar_t *__restrict__ viewmats1, // [C, 4, 4] optional for rolling shutter
    const scalar_t *__restrict__ Ks,        // [C, 3, 3]
    const CameraModelType camera_model_type,
    // uncented transform
    const UnscentedTransformParameters ut_params,    
    const ShutterType rs_type,
    const scalar_t *__restrict__ radial_coeffs, // [C, 6] or [C, 4] optional
    const scalar_t *__restrict__ tangential_coeffs, // [C, 2] optional
    const scalar_t *__restrict__ thin_prism_coeffs, // [C, 2] optional
    // intersections
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    scalar_t
        *__restrict__ render_colors, // [C, image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas, // [C, image_height, image_width, 1]
    int32_t *__restrict__ last_ids        // [C, image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t cid = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += cid * tile_height * tile_width;
    render_colors += cid * image_height * image_width * CDIM;
    render_alphas += cid * image_height * image_width;
    last_ids += cid * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += cid * CDIM;
    }
    if (masks != nullptr) {
        masks += cid * tile_height * tile_width;
    }

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // Create rolling shutter parameter
    auto rs_params = RollingShutterParameters(
        viewmats0 + cid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + cid * 16
    );
    // shift pointers to the current camera. note that glm is colume-major.
    const vec2 focal_length = {Ks[cid * 9 + 0], Ks[cid * 9 + 4]};
    const vec2 principal_point = {Ks[cid * 9 + 2], Ks[cid * 9 + 5]};
    
    // Create ray from pixel
    WorldRay ray;
    if (camera_model_type == CameraModelType::PINHOLE) {
        if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
            PerfectPinholeCameraModel::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = { principal_point.x, principal_point.y };
            cm_params.focal_length = { focal_length.x, focal_length.y };
            PerfectPinholeCameraModel camera_model(cm_params);
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        } else {
            OpenCVPinholeCameraModel<>::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = { principal_point.x, principal_point.y };
            cm_params.focal_length = { focal_length.x, focal_length.y };
            if (radial_coeffs != nullptr) {
                cm_params.radial_coeffs = make_array<float, 6>(radial_coeffs + cid * 6);
            }
            if (tangential_coeffs != nullptr) {
                cm_params.tangential_coeffs = make_array<float, 2>(tangential_coeffs + cid * 2);
            }
            if (thin_prism_coeffs != nullptr) {
                cm_params.thin_prism_coeffs = make_array<float, 4>(thin_prism_coeffs + cid * 4);
            }
            OpenCVPinholeCameraModel camera_model(cm_params);
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        }
    } else if (camera_model_type == CameraModelType::FISHEYE) {
        OpenCVFisheyeCameraModel<>::Parameters cm_params = {};
        cm_params.resolution = {image_width, image_height};
        cm_params.shutter_type = rs_type;
        cm_params.principal_point = { principal_point.x, principal_point.y };
        cm_params.focal_length = { focal_length.x, focal_length.y };
        if (radial_coeffs != nullptr) {
            cm_params.radial_coeffs = make_array<float, 4>(radial_coeffs + cid * 4);
        }
        OpenCVFisheyeCameraModel camera_model(cm_params);
        ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
    } else {
        // should never reach here
        assert(false);
        return;
    }
    vec3 ray_d = ray.ray_dir;
    vec3 ray_o = ray.ray_org;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (cid == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec4 *xyz_opacity_batch =
        reinterpret_cast<vec4 *>(&id_batch[block_size]); // [block_size]
    mat3 *iscl_rot_batch =
        reinterpret_cast<mat3 *>(&xyz_opacity_batch[block_size]); // [block_size]
    
    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    float T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    float pix_out[CDIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            // TODO: only support 1 camera for now so it is ok to abuse the index.
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec3 xyz = means[g];
            const float opac = opacities[g];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
            
            const vec4 quat = quats[g];
            vec3 scale = scales[g];
            
            mat3 R = quat_to_rotmat(quat);
            mat3 S = mat3(
                1.0f / scale[0],
                0.f,
                0.f,
                0.f,
                1.0f / scale[1],
                0.f,
                0.f,
                0.f,
                1.0f / scale[2]
            );
            mat3 iscl_rot = S * glm::transpose(R);
            iscl_rot_batch[tr] = iscl_rot;
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec4 xyz_opac = xyz_opacity_batch[t];
            const float opac = xyz_opac[3];
            const vec3 xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};            
            const mat3 iscl_rot = iscl_rot_batch[t];

            const vec3 gro = iscl_rot * (ray_o - xyz);
            const vec3 grd = safe_normalize(iscl_rot * ray_d);
            const vec3 gcrod = glm::cross(grd, gro);
            const float grayDist = glm::dot(gcrod, gcrod);
            const float power = -0.5f * grayDist;

            float alpha = min(0.999f, opac * __expf(power));
            if (alpha < 1.f / 255.f) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward
        // pass and it can be very small and causing large diff in gradients
        // with float32. However, double precision makes the backward pass 1.5x
        // slower so we stick with float for now.
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means, // [N, 3]
    const at::Tensor quats, // [N, 4]
    const at::Tensor scales, // [N, 3]
    const at::Tensor colors,    // [C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [C, N]  or [nnz]
    const at::optional<at::Tensor> backgrounds, // [C, channels]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0,             // [C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                   // [C, 3, 3]
    const CameraModelType camera_model,
    // uncented transform
    const UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs, // [C, 6] or [C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [C, 2] optional
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders, // [C, image_height, image_width, channels]
    at::Tensor alphas,  // [C, image_height, image_width]
    at::Tensor last_ids // [C, image_height, image_width]
) {
    // Note: quats need to be normalized before passing in.

    bool packed = opacities.dim() == 1;
    assert (packed == false); // only support non-packed for now

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means.size(0);   // number of gaussians
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size * 
        (sizeof(int32_t) + sizeof(vec4) + sizeof(mat3));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
        rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM, float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size
    ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec3 *>(means.data_ptr<float>()),
            reinterpret_cast<vec4 *>(quats.data_ptr<float>()),
            reinterpret_cast<vec3 *>(scales.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            // camera model
            viewmats0.data_ptr<float>(),
            viewmats1.has_value() ? viewmats1.value().data_ptr<float>()
                                  : nullptr,
            Ks.data_ptr<float>(),
            camera_model,
            // uncented transform
            ut_params,
            rs_type,
            radial_coeffs.has_value() ? radial_coeffs.value().data_ptr<float>()
                                      : nullptr,
            tangential_coeffs.has_value()
                ? tangential_coeffs.value().data_ptr<float>()
                : nullptr,
            thin_prism_coeffs.has_value()
                ? thin_prism_coeffs.value().data_ptr<float>()
                : nullptr,
            // intersections
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>()
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM>( \
        const at::Tensor means,                                                \
        const at::Tensor quats,                                                \
        const at::Tensor scales,                                               \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        const uint32_t image_width,                                            \
        const uint32_t image_height,                                           \
        const uint32_t tile_size,                                              \
        const at::Tensor viewmats0,                                            \
        const at::optional<at::Tensor> viewmats1,                              \
        const at::Tensor Ks,                                                   \
        const CameraModelType camera_model,                                    \
        const UnscentedTransformParameters ut_params,                         \
        const ShutterType rs_type,                                             \
        const at::optional<at::Tensor> radial_coeffs,                         \
        const at::optional<at::Tensor> tangential_coeffs,                     \
        const at::optional<at::Tensor> thin_prism_coeffs,                     \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const at::Tensor renders,                                              \
        const at::Tensor alphas,                                               \
        const at::Tensor last_ids                                               \
    );                                                                        

__INS__(3)
#undef __INS__


} // namespace gsplat
