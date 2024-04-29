#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>

#define MAX_BLOCK_SIZE (16 * 16)
#define N_THREADS 256

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                 \
    CHECK_CUDA(x);                                                                     \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten)                                                             \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
#define CUB_WRAPPER(func, ...)                                                         \
    do {                                                                               \
        size_t temp_storage_bytes = 0;                                                 \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                                \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();           \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);            \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);                     \
    } while (false)

std::tuple<torch::Tensor, torch::Tensor>
quat_scale_to_covar_perci_fwd_tensor(const torch::Tensor &quats,  // [N, 4]
                                     const torch::Tensor &scales, // [N, 3]
                                     const bool compute_covar, const bool compute_perci,
                                     const bool triu);

std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_perci_bwd_tensor(
    const torch::Tensor &quats,                  // [N, 4]
    const torch::Tensor &scales,                 // [N, 3]
    const at::optional<torch::Tensor> &v_covars, // [N, 3, 3]
    const at::optional<torch::Tensor> &v_percis, // [N, 3, 3]
    const bool triu);

std::tuple<torch::Tensor, torch::Tensor>
persp_proj_fwd_tensor(const torch::Tensor &means,  // [C, N, 3]
                      const torch::Tensor &covars, // [C, N, 3, 3]
                      const torch::Tensor &Ks,     // [C, 3, 3]
                      const int width, const int height);

std::tuple<torch::Tensor, torch::Tensor>
persp_proj_bwd_tensor(const torch::Tensor &means,  // [C, N, 3]
                      const torch::Tensor &covars, // [C, N, 3, 3]
                      const torch::Tensor &Ks,     // [C, 3, 3]
                      const int width, const int height,
                      const torch::Tensor &v_means2d, // [C, N, 2]
                      const torch::Tensor &v_covars2d // [C, N, 2, 2]
);

std::tuple<torch::Tensor, torch::Tensor>
world_to_cam_fwd_tensor(const torch::Tensor &means,   // [N, 3]
                        const torch::Tensor &covars,  // [N, 3, 3]
                        const torch::Tensor &viewmats // [C, 4, 4]
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_cam_bwd_tensor(const torch::Tensor &means,                    // [N, 3]
                        const torch::Tensor &covars,                   // [N, 3, 3]
                        const torch::Tensor &viewmats,                 // [C, 4, 4]
                        const at::optional<torch::Tensor> &v_means_c,  // [C, N, 3]
                        const at::optional<torch::Tensor> &v_covars_c, // [C, N, 3, 3]
                        const bool means_requires_grad, const bool covars_requires_grad,
                        const bool viewmats_requires_grad);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
projection_fwd_tensor(const torch::Tensor &means,    // [N, 3]
                      const torch::Tensor &covars,   // [N, 6]
                      const torch::Tensor &viewmats, // [C, 4, 4]
                      const torch::Tensor &Ks,       // [C, 3, 3]
                      const int image_width, const int image_height, const float eps2d,
                      const float near_plane);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> projection_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &covars,   // [N, 6]
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    const int image_width, const int image_height,
    // fwd outputs
    const torch::Tensor &radii,  // [C, N]
    const torch::Tensor &conics, // [C, N, 3]
    // grad outputs
    const torch::Tensor &v_means2d, // [C, N, 2]
    const torch::Tensor &v_depths,  // [C, N]
    const torch::Tensor &v_conics,  // [C, N, 3]
    const bool viewmats_requires_grad);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_tiles_tensor(const torch::Tensor &means2d, // [C, N, 2]
                   const torch::Tensor &radii,   // [C, N]
                   const torch::Tensor &depths,  // [C, N]
                   const int tile_size, const int tile_width, const int tile_height,
                   const bool sort);

torch::Tensor isect_offset_encode_tensor(const torch::Tensor &isect_ids, // [n_isects]
                                         const int C, const int tile_width,
                                         const int tile_height);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_to_pixels_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &colors,    // [C, N, 3]
    const torch::Tensor &opacities, // [N]
    // image size
    const int image_width, const int image_height, const int tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &gauss_ids     // [n_isects]
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &colors,    // [C, N, 3]
    const torch::Tensor &opacities, // [N]
    // image size
    const int image_width, const int image_height, const int tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &gauss_ids,    // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas  // [C, image_height, image_width, 1]
);