#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <uint32_t COLOR_DIM, bool GEO, typename S>
__global__ void rasterize_to_pixels_fwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,

    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,        // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,     // [C, N] or [nnz]
    const S *__restrict__ ray_ts,        // [C, N] or [nnz]
    const vec2<S> *__restrict__ ray_planes,    // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ normals,       // [C, N, 3] or [nnz, 3]
    const S *__restrict__ backgrounds,   // [C, COLOR_DIM]
    const bool *__restrict__ masks,      // [C, tile_height, tile_width]
    const S *__restrict__ Ks,            // [C, 9]
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    const uint32_t tile_width, const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    S *__restrict__ render_colors, // [C, image_height, image_width, COLOR_DIM]
    S *__restrict__ render_alphas, // [C, image_height, image_width, 1]
    S *__restrict__ render_depths, // [C, image_height, image_width, 1]
    S *__restrict__ median_depths, // [C, image_height, image_width, 1]
    S *__restrict__ render_normals,// [C, image_height, image_width, 3]
    int32_t *__restrict__ median_ids, // [C, image_height, image_width]
    int32_t *__restrict__ last_ids // [C, image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_colors += camera_id * image_height * image_width * COLOR_DIM;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    S px = (S)j + 0.5f;
    S py = (S)i + 0.5f;
    int32_t pix_id = i * image_width + j;
    S ln;
    if constexpr (GEO)
    {
        Ks += camera_id * 9;
        S fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
        vec3<S> pixnf = {(px - cx) / fx, (py - cy) / fy, 1};
        ln = glm::length(pixnf);
    }


    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] = backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3<S> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]); // [block_size]
    S *ray_t_batch;
    vec2<S> *ray_plane_batch;
    vec3<S> *normal_batch;
    if constexpr (GEO)
    {
        ray_t_batch = 
            reinterpret_cast<float *>(&conic_batch[block_size]); // [block_size]
        ray_plane_batch =
            reinterpret_cast<vec2<float> *>(&ray_t_batch[block_size]); // [block_size]
        normal_batch =
            reinterpret_cast<vec3<float> *>(&ray_plane_batch[block_size]); // [block_size]
    }
    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x slower
    // so we stick with float for now.
    S T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;
    uint32_t median_idx = -1;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    S pix_out[COLOR_DIM] = {0.f};
    S t_out = 0.f;
    S normal_out[3] = {0.f};
    S t_median = 0.f;
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
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            if constexpr (GEO)
            {
                ray_t_batch[tr] = ray_ts[g];
                ray_plane_batch[tr] = ray_planes[g];
                normal_batch[tr] = normals[g];
            }
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3<S> conic = conic_batch[t];
            const vec3<S> xy_opac = xy_opacity_batch[t];
            const S opac = xy_opac.z;
            const vec2<S> delta = {xy_opac.x - px, xy_opac.y - py};
            const S sigma =
                0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                conic.y * delta.x * delta.y;
            S alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const S next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const S vis = alpha * T;
            const S *c_ptr = colors + g * COLOR_DIM;
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            if constexpr (GEO)
            {
                const S ray_t = ray_t_batch[t];
                const vec2<S> ray_plane = ray_plane_batch[t];
                const vec3<S> normal = normal_batch[t];
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < 3; ++k) {
                    normal_out[k] += normal[k] * vis;
                }

                S t_opt = ray_t + glm::dot(delta, ray_plane);
                t_out += t_opt * vis;
                if (T > 0.5)
                {
                    median_idx = batch_start + t;
                    t_median = t_opt;
                }
                // printf("%f %f %f %f %f %f\n",depth_out,t_opt,ray_t,ln,ray_plane.x,ray_plane.y);
            }
            
            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward pass and
        // it can be very small and causing large diff in gradients with float32.
        // However, double precision makes the backward pass 1.5x slower so we stick
        // with float for now.
        render_alphas[pix_id] = 1.0f - T;
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? pix_out[k] : (pix_out[k] + T * backgrounds[k]);
        }
        if constexpr (GEO)
        {
            render_depths[pix_id] = t_out / ln;
            median_depths[pix_id] = t_median / ln;
            median_ids[pix_id] = median_idx;
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < 3; ++k){
                render_normals[pix_id * 3 + k] = normal_out[k];
            }
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <class T, uint32_t CDIM>
T call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,    // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,     // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,     // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities,  // [C, N]  or [nnz]
    const torch::Tensor &ray_ts,     // [C, N] or [nnz]
    const torch::Tensor &ray_planes, // [C, N, 2] or [nnz, 2]
    const torch::Tensor &normals,    // [C, N, 3] or [nnz, 3]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    const torch::Tensor &Ks,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    bool packed = means2d.dim() == 2;

    constexpr unsigned int output_size = std::tuple_size_v<T>;
    constexpr bool GEO = output_size > 3;
    if constexpr (GEO)
    {
        CHECK_INPUT(ray_ts);
        CHECK_INPUT(ray_planes);
        CHECK_INPUT(normals);
    }

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t channels = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor renders = torch::empty({C, image_height, image_width, channels},
                                         means2d.options().dtype(torch::kFloat32));
    torch::Tensor alphas = torch::empty({C, image_height, image_width, 1},
                                        means2d.options().dtype(torch::kFloat32));
    torch::Tensor expected_depths = GEO ? torch::empty({C, image_height, image_width, 1},
                                            means2d.options().dtype(torch::kFloat32)) : torch::Tensor();
    torch::Tensor median_depths = GEO ? torch::empty({C, image_height, image_width, 1},
                                            means2d.options().dtype(torch::kFloat32)) : torch::Tensor();
    torch::Tensor expected_normals = GEO ? torch::empty({C, image_height, image_width, 3},
                                        means2d.options().dtype(torch::kFloat32)) : torch::Tensor();
    torch::Tensor median_ids = torch::empty({C, image_height, image_width},
                                          means2d.options().dtype(torch::kInt32));
    torch::Tensor last_ids = torch::empty({C, image_height, image_width},
                                          means2d.options().dtype(torch::kInt32));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem = 
        tile_size * tile_size *
        (GEO ? sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>) + sizeof(float) + sizeof(vec2<float>) + sizeof(vec3<float>)
                :sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>));

    // TODO: an optimization can be done by passing the actual number of channels into
    // the kernel functions and avoid necessary global memory writes. This requires
    // moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<CDIM, GEO, float>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                 " bytes), try lowering tile_size.");
    }
    
    rasterize_to_pixels_fwd_kernel<CDIM, GEO, float>
        <<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(), opacities.data_ptr<float>(),
            GEO ? ray_ts.data_ptr<float>() : nullptr, 
            GEO ? reinterpret_cast<vec2<float> *>(ray_planes.data_ptr<float>()) : nullptr, 
            GEO ? reinterpret_cast<vec3<float> *>(normals.data_ptr<float>()) : nullptr,
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            GEO ? Ks.data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(), alphas.data_ptr<float>(),
            GEO ? expected_depths.data_ptr<float>() : nullptr, 
            GEO ? median_depths.data_ptr<float>() : nullptr, 
            GEO ? expected_normals.data_ptr<float>() : nullptr,
            GEO ? median_ids.data_ptr<int32_t>() : nullptr,
            last_ids.data_ptr<int32_t>());
    
    if constexpr (GEO)
        return std::make_tuple(renders, alphas, expected_depths, median_depths, expected_normals, median_ids, last_ids);
    else
        return std::make_tuple(renders, alphas, last_ids);
}

template<class T>
T rasterize_to_pixels_fwd(
    // Gaussian parameters
    const torch::Tensor &means2d,    // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,     // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,     // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities,  // [C, N]  or [nnz]
    const torch::Tensor &ray_ts,     // [C, N] or [nnz]
    const torch::Tensor &ray_planes, // [C, N, 2] or [nnz, 2]
    const torch::Tensor &normals,    // [C, N, 3] or [nnz, 3]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    const torch::Tensor &Ks,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
){
    CHECK_INPUT(colors);
    uint32_t channels = colors.size(-1);

#define __GS__CALL_(N)                                                                 \
    case N:                                                                            \
        return call_kernel_with_dim<T, N>(means2d, conics, colors, opacities,             \
                                       ray_ts, ray_planes, normals,                    \
                                       backgrounds, masks, image_width, image_height, \
                                       tile_size, Ks, tile_offsets, flatten_ids);

    // TODO: an optimization can be done by passing the actual number of channels into
    // the kernel functions and avoid necessary global memory writes. This requires
    // moving the channel padding from python to C side.
    switch (channels) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_wo_depth_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,    // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,     // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,     // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities,  // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
){
    torch::Tensor ray_ts;
    torch::Tensor ray_planes;
    torch::Tensor normals;
    torch::Tensor Ks;
    return rasterize_to_pixels_fwd<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
                                    (means2d, conics, colors, opacities, ray_ts, ray_planes, normals, 
                                        backgrounds, masks, image_width, image_height, tile_size, 
                                        Ks, tile_offsets, flatten_ids);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_w_depth_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,    // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,     // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,     // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities,  // [C, N]  or [nnz]
    const torch::Tensor &ray_ts,     // [C, N] or [nnz]
    const torch::Tensor &ray_planes, // [C, N, 2] or [nnz, 2]
    const torch::Tensor &normals,    // [C, N, 3] or [nnz, 3]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    const torch::Tensor &Ks,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
){
    return rasterize_to_pixels_fwd<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
                                    (means2d, conics, colors, opacities, ray_ts, ray_planes, normals, 
                                        backgrounds, masks, image_width, image_height, tile_size, 
                                        Ks, tile_offsets, flatten_ids);
}