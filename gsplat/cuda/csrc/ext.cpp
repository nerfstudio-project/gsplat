#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>

#include "common_host.h"
#include "proj_naive.h"
#include "proj_fused.h"
#include "proj_fused_packed.h"
#include "sh.h"


/****************************************************************************
 * Naive Projection
 ****************************************************************************/

std::tuple<torch::Tensor, torch::Tensor> proj_naive_fwd(
    const torch::Tensor means,  // [C, N, 3]
    const torch::Tensor covars, // [C, N, 3, 3]
    const torch::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor covars2d = torch::empty({C, N, 2, 2}, covars.options());

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    uint32_t n_elements = C * N;
    uint32_t shmem_size = 0;
    proj_naive_fwd_launcher(
        shmem_size,
        stream,
        n_elements,
        // args
        C, 
        N,
        means.data_ptr<float>(),
        covars.data_ptr<float>(),
        Ks.data_ptr<float>(),
        width,
        height,
        camera_model,
        means2d.data_ptr<float>(),
        covars2d.data_ptr<float>()
    );
    return std::make_tuple(means2d, covars2d);
}

std::tuple<torch::Tensor, torch::Tensor> proj_naive_bwd(
    const torch::Tensor means,  // [C, N, 3]
    const torch::Tensor covars, // [C, N, 3, 3]
    const torch::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const torch::Tensor v_means2d, // [C, N, 2]
    const torch::Tensor v_covars2d // [C, N, 2, 2]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_covars2d);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    torch::Tensor v_means = torch::empty({C, N, 3}, means.options());
    torch::Tensor v_covars = torch::empty({C, N, 3, 3}, means.options());

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    uint32_t n_elements = C * N;
    uint32_t shmem_size = 0;
    proj_naive_bwd_launcher(
        shmem_size,
        stream,
        n_elements,
        // args
        C, 
        N,
        means.data_ptr<float>(),
        covars.data_ptr<float>(),
        Ks.data_ptr<float>(),
        width,
        height,
        camera_model,
        v_means2d.data_ptr<float>(),
        v_covars2d.data_ptr<float>(),
        v_means.data_ptr<float>(),
        v_covars.data_ptr<float>()
    );
    return std::make_tuple(v_means, v_covars);
}



std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
proj_fused_fwd(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6] optional
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras

    torch::Tensor radii =
        torch::empty({C, N}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::empty({C, N}, means.options());
    torch::Tensor conics = torch::empty({C, N, 3}, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({C, N}, means.options());
    }
    
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    uint32_t n_elements = C * N;
    uint32_t shmem_size = 0;
    proj_fused_fwd_launcher(
        shmem_size,
        stream,
        n_elements,
        // args
        C,
        N,
        means.data_ptr<float>(),
        covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
        quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
        scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
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
        means2d.data_ptr<float>(),
        depths.data_ptr<float>(),
        conics.data_ptr<float>(),
        calc_compensations ? compensations.data_ptr<float>() : nullptr
    );
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}


/****************************************************************************
 * Fused Projection
 ****************************************************************************/


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
proj_fused_bwd(
    // fwd inputs
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6] optional
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const torch::Tensor &radii,                       // [C, N]
    const torch::Tensor &conics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &compensations, // [C, N] optional
    // grad outputs
    const torch::Tensor &v_means2d,                     // [C, N, 2]
    const torch::Tensor &v_depths,                      // [C, N]
    const torch::Tensor &v_conics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &v_compensations, // [C, N] optional
    const bool viewmats_requires_grad
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);
    if (compensations.has_value()) {
        CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_covars, v_quats, v_scales; // optional
    if (covars.has_value()) {
        v_covars = torch::zeros_like(covars.value());
    } else {
        v_quats = torch::zeros_like(quats.value());
        v_scales = torch::zeros_like(scales.value());
    }
    torch::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros_like(viewmats);
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    uint32_t n_elements = C * N;
    uint32_t shmem_size = 0;
    proj_fused_bwd_launcher(
        shmem_size,
        stream,
        n_elements,
        // args
        C,
        N,
        means.data_ptr<float>(),
        covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
        covars.has_value() ? nullptr : quats.value().data_ptr<float>(),
        covars.has_value() ? nullptr : scales.value().data_ptr<float>(),
        viewmats.data_ptr<float>(),
        Ks.data_ptr<float>(),
        image_width,
        image_height,
        eps2d,
        camera_model,
        radii.data_ptr<int32_t>(),
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
        v_means.data_ptr<float>(),
        covars.has_value() ? v_covars.data_ptr<float>() : nullptr,
        covars.has_value() ? nullptr : v_quats.data_ptr<float>(),
        covars.has_value() ? nullptr : v_scales.data_ptr<float>(),
        viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
    );
    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}


/****************************************************************************
 * Fused Projection (Packed)
 ****************************************************************************/


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
proj_fused_packed_fwd(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6]
    const at::optional<torch::Tensor> &quats,  // [N, 3]
    const at::optional<torch::Tensor> &scales, // [N, 3]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    auto opt = means.options().dtype(torch::kInt32);

    uint32_t nrows = C;
    uint32_t ncols = N;
    uint32_t blocks_per_row = (ncols + N_THREADS - 1) / N_THREADS;

    dim3 threads = {N_THREADS, 1, 1};
    // limit on the number of blocks: [2**31 - 1, 65535, 65535]
    dim3 blocks = {blocks_per_row, nrows, 1};

    // first pass
    int32_t nnz;
    torch::Tensor block_accum;
    if (C && N) {
        torch::Tensor block_cnts = torch::empty({nrows * blocks_per_row}, opt);
        proj_fused_packed_fwd_launcher(
            0,
            stream,
            blocks,
            threads,
            // args
            C,
            N,
            means.data_ptr<float>(),
            covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
            quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
            scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
            viewmats.data_ptr<float>(),
            Ks.data_ptr<float>(),
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            nullptr,
            camera_model,
            block_cnts.data_ptr<int32_t>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr
        );
        block_accum = torch::cumsum(block_cnts, 0, torch::kInt32);
        nnz = block_accum[-1].item<int32_t>();
    } else {
        nnz = 0;
    }

    // second pass
    torch::Tensor indptr = torch::empty({C + 1}, opt);
    torch::Tensor camera_ids = torch::empty({nnz}, opt.dtype(torch::kInt64));
    torch::Tensor gaussian_ids = torch::empty({nnz}, opt.dtype(torch::kInt64));
    torch::Tensor radii =
        torch::empty({nnz}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({nnz, 2}, means.options());
    torch::Tensor depths = torch::empty({nnz}, means.options());
    torch::Tensor conics = torch::empty({nnz, 3}, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({nnz}, means.options());
    }

    if (nnz) {
        proj_fused_packed_fwd_launcher(
            0,
            stream,
            blocks,
            threads,
            // args
            C,
            N,
            means.data_ptr<float>(),
            covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
            quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
            scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
            viewmats.data_ptr<float>(),
            Ks.data_ptr<float>(),
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            block_accum.data_ptr<int32_t>(),
            camera_model,
            nullptr,
            indptr.data_ptr<int32_t>(),
            camera_ids.data_ptr<int64_t>(),
            gaussian_ids.data_ptr<int64_t>(),
            radii.data_ptr<int32_t>(),
            means2d.data_ptr<float>(),
            depths.data_ptr<float>(),
            conics.data_ptr<float>(),
            calc_compensations ? compensations.data_ptr<float>() : nullptr
        );
    } else {
        indptr.fill_(0);
    }

    return std::make_tuple(
        indptr,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        compensations
    );
}




std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
proj_fused_packed_bwd(
    // fwd inputs
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6]
    const at::optional<torch::Tensor> &quats,  // [N, 4]
    const at::optional<torch::Tensor> &scales, // [N, 3]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const torch::Tensor &camera_ids,                  // [nnz]
    const torch::Tensor &gaussian_ids,                // [nnz]
    const torch::Tensor &conics,                      // [nnz, 3]
    const at::optional<torch::Tensor> &compensations, // [nnz] optional
    // grad outputs
    const torch::Tensor &v_means2d,                     // [nnz, 2]
    const torch::Tensor &v_depths,                      // [nnz]
    const torch::Tensor &v_conics,                      // [nnz, 3]
    const at::optional<torch::Tensor> &v_compensations, // [nnz] optional
    const bool viewmats_requires_grad,
    const bool sparse_grad
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(camera_ids);
    CHECK_INPUT(gaussian_ids);
    CHECK_INPUT(conics);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);
    if (compensations.has_value()) {
        CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    uint32_t nnz = camera_ids.size(0);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means, v_covars, v_quats, v_scales, v_viewmats;
    if (sparse_grad) {
        v_means = torch::zeros({nnz, 3}, means.options());
        if (covars.has_value()) {
            v_covars = torch::zeros({nnz, 6}, covars.value().options());
        } else {
            v_quats = torch::zeros({nnz, 4}, quats.value().options());
            v_scales = torch::zeros({nnz, 3}, scales.value().options());
        }
        if (viewmats_requires_grad) {
            v_viewmats = torch::zeros({C, 4, 4}, viewmats.options());
        }
    } else {
        v_means = torch::zeros_like(means);
        if (covars.has_value()) {
            v_covars = torch::zeros_like(covars.value());
        } else {
            v_quats = torch::zeros_like(quats.value());
            v_scales = torch::zeros_like(scales.value());
        }
        if (viewmats_requires_grad) {
            v_viewmats = torch::zeros_like(viewmats);
        }
    }

    proj_fused_packed_bwd_launcher(
        0,
        stream,
        nnz,
        // args
        C,
        N,
        nnz,
        means.data_ptr<float>(),
        covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
        covars.has_value() ? nullptr : quats.value().data_ptr<float>(),
        covars.has_value() ? nullptr : scales.value().data_ptr<float>(),
        viewmats.data_ptr<float>(),
        Ks.data_ptr<float>(),
        image_width,
        image_height,
        eps2d,
        camera_model,
        camera_ids.data_ptr<int64_t>(),
        gaussian_ids.data_ptr<int64_t>(),
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
        sparse_grad,
        v_means.data_ptr<float>(),
        covars.has_value() ? v_covars.data_ptr<float>() : nullptr,
        covars.has_value() ? nullptr : v_quats.data_ptr<float>(),
        covars.has_value() ? nullptr : v_scales.data_ptr<float>(),
        viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
    );
    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}

/****************************************************************************
 * Spherical Harmonics
 ****************************************************************************/

torch::Tensor sh_fwd(
    const uint32_t degrees_to_use,
    const torch::Tensor &dirs,              // [..., 3]
    const torch::Tensor &coeffs,            // [..., K, 3]
    const at::optional<torch::Tensor> masks // [...]
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(coeffs);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;
    torch::Tensor colors = torch::empty_like(dirs); // [..., 3]

    // parallelize over N * 3
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    uint32_t n_elements = N;
    uint32_t shmem_size = 0;
    sh_fwd_launcher(
        shmem_size,
        stream,
        n_elements,
        // args
        N,
        K,
        degrees_to_use,
        reinterpret_cast<vec3 *>(dirs.data_ptr<float>()),
        coeffs.data_ptr<float>(),
        masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
        colors.data_ptr<float>()
    );
    return colors; // [..., 3]
}


std::tuple<torch::Tensor, torch::Tensor> sh_bwd(
    const uint32_t K,
    const uint32_t degrees_to_use,
    const torch::Tensor &dirs,               // [..., 3]
    const torch::Tensor &coeffs,             // [..., K, 3]
    const at::optional<torch::Tensor> masks, // [...]
    const torch::Tensor &v_colors,           // [..., 3]
    bool compute_v_dirs
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(coeffs);
    CHECK_INPUT(v_colors);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(v_colors.size(-1) == 3, "v_colors must have last dimension 3");
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
    const uint32_t N = dirs.numel() / 3;

    torch::Tensor v_coeffs = torch::zeros_like(coeffs);
    torch::Tensor v_dirs;
    if (compute_v_dirs) {
        v_dirs = torch::zeros_like(dirs);
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    uint32_t n_elements = N;
    uint32_t shmem_size = 0;
    sh_bwd_launcher(
        shmem_size,
        stream,
        n_elements,
        // args
        N,
        K,
        degrees_to_use,
        reinterpret_cast<vec3 *>(dirs.data_ptr<float>()),
        coeffs.data_ptr<float>(),
        masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
        v_colors.data_ptr<float>(),
        v_coeffs.data_ptr<float>(),
        compute_v_dirs ? v_dirs.data_ptr<float>() : nullptr
    );
    return std::make_tuple(v_coeffs, v_dirs); // [..., K, 3], [..., 3]
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("proj_naive_fwd", proj_naive_fwd);
    m.def("proj_naive_bwd", proj_naive_bwd);

    m.def("proj_fused_fwd", proj_fused_fwd);
    m.def("proj_fused_bwd", proj_fused_bwd);

    m.def("proj_fused_packed_fwd", proj_fused_packed_fwd);
    m.def("proj_fused_packed_bwd", proj_fused_packed_bwd);

    m.def("sh_fwd", sh_fwd);
    m.def("sh_bwd", sh_bwd);
}
