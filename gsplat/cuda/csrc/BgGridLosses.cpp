/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "GSplatBuildConfig.h"

#if GSPLAT_BUILD_LOSSES

#    include <ATen/Functions.h>
#    include <ATen/TensorUtils.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAGuard.h>
#    include <torch/library.h>

#    include "BgGridLosses.h"
#    include "Common.h"

namespace gsplat
{
// Output tensors are pre-allocated on the Python side — the C++ op takes
// mutable Tensors and returns void so buffer lifetime stays explicit and
// torch's caching allocator can reuse across training steps.

void bg_grid_losses_fwd(
    const at::optional<at::Tensor> &bg_tex,
    int64_t bg_tex_depth,
    const at::optional<at::Tensor> &grids_camera,
    const at::optional<at::Tensor> &grids_frame,
    double bg_tex_factor,
    double grid_drift_camera_factor,
    double grid_drift_frame_factor,
    double grid_camera_tv_factor,
    double grid_frame_tv_factor,
    at::Tensor bg_tex_loss,
    at::Tensor grids_drift_loss,
    at::Tensor grid_camera_tv_loss,
    at::Tensor grid_frame_tv_loss
)
{
    // Pick a reference CUDA device from the first provided input and hold one
    // guard alive through the launch. The per-block DEVICE_GUARDs used before
    // went out of scope before the kernel ran, so the launch could pick up the
    // wrong current device.
    const at::Tensor *ref = nullptr;
    if(bg_tex.has_value())
    {
        ref = &bg_tex.value();
    }
    else if(grids_camera.has_value())
    {
        ref = &grids_camera.value();
    }
    else if(grids_frame.has_value())
    {
        ref = &grids_frame.value();
    }
    TORCH_CHECK(
        ref != nullptr,
        "bg_grid_losses_fwd: at least one of bg_tex / grids_camera / "
        "grids_frame must be provided"
    );
    const at::cuda::OptionalCUDAGuard device_guard(device_of(*ref));
    const auto dev = ref->device();

    CHECK_INPUT(bg_tex_loss);
    CHECK_INPUT(grids_drift_loss);
    CHECK_INPUT(grid_camera_tv_loss);
    CHECK_INPUT(grid_frame_tv_loss);

    int B_tex = 0, D_tex = 0, H_tex = 0, W_tex = 0, C_tex = 0;
    int B_gc = 0, D_gc = 0, H_gc = 0, W_gc = 0;
    int B_gf = 0, D_gf = 0, H_gf = 0, W_gf = 0;

    const at::Tensor *bg_ptr = nullptr;
    const at::Tensor *gc_ptr = nullptr;
    const at::Tensor *gf_ptr = nullptr;

    if(bg_tex.has_value())
    {
        const auto &t = bg_tex.value();
        CHECK_INPUT(t);
        TORCH_CHECK(t.device() == dev, "bg_tex must be on the same device as the other inputs");
        TORCH_CHECK(t.dim() == 4 && t.scalar_type() == at::kFloat, "bg_tex must be float32 of shape [B*D, H, W, C]");
        const int BD = t.size(0);
        TORCH_CHECK(bg_tex_depth >= 1, "bg_tex_depth must be >= 1 (1 for planar, 6 for cubemap)");
        TORCH_CHECK(BD % bg_tex_depth == 0, "bg_tex.size(0) must be divisible by bg_tex_depth");
        D_tex  = static_cast<int>(bg_tex_depth);
        B_tex  = BD / D_tex;
        H_tex  = t.size(1);
        W_tex  = t.size(2);
        C_tex  = t.size(3);
        bg_ptr = &t;
    }
    if(grids_camera.has_value())
    {
        const auto &t = grids_camera.value();
        CHECK_INPUT(t);
        TORCH_CHECK(t.device() == dev, "grids_camera must be on the same device as the other inputs");
        TORCH_CHECK(
            t.dim() == 4 && t.scalar_type() == at::kFloat,
            "grids_camera must be float32 of shape "
            "[B*12, D, H, W]"
        );
        const int B12 = t.size(0);
        TORCH_CHECK(
            B12 % LOSSES_GRID_NUM_CHANNELS == 0, "grids_camera.size(0) must be divisible by ", LOSSES_GRID_NUM_CHANNELS
        );
        B_gc   = B12 / LOSSES_GRID_NUM_CHANNELS;
        D_gc   = t.size(1);
        H_gc   = t.size(2);
        W_gc   = t.size(3);
        gc_ptr = &t;
    }
    if(grids_frame.has_value())
    {
        const auto &t = grids_frame.value();
        CHECK_INPUT(t);
        TORCH_CHECK(t.device() == dev, "grids_frame must be on the same device as the other inputs");
        TORCH_CHECK(
            t.dim() == 4 && t.scalar_type() == at::kFloat,
            "grids_frame must be float32 of shape "
            "[B*12, D, H, W]"
        );
        const int B12 = t.size(0);
        TORCH_CHECK(
            B12 % LOSSES_GRID_NUM_CHANNELS == 0, "grids_frame.size(0) must be divisible by ", LOSSES_GRID_NUM_CHANNELS
        );
        B_gf   = B12 / LOSSES_GRID_NUM_CHANNELS;
        D_gf   = t.size(1);
        H_gf   = t.size(2);
        W_gf   = t.size(3);
        gf_ptr = &t;
    }

    // Output buffers must exactly match what the kernel writes; a direct
    // torch.ops.gsplat.bg_grid_losses_fwd caller could otherwise pass an
    // undersized or wrong-dtype/-device buffer and corrupt memory.
    const int64_t numel_bg = static_cast<int64_t>(B_tex) * D_tex * H_tex * W_tex * C_tex;
    const int64_t numel_gc = static_cast<int64_t>(B_gc) * D_gc * H_gc * W_gc;
    const int64_t numel_gf = static_cast<int64_t>(B_gf) * D_gf * H_gf * W_gf;
    auto check_out         = [&](const at::Tensor &o, int64_t n, const char *name)
    {
        TORCH_CHECK(o.scalar_type() == at::kFloat, name, " must be float32");
        TORCH_CHECK(o.device() == dev, name, " must be on the input device");
        TORCH_CHECK(o.numel() == n, name, " numel mismatch: expected ", n, ", got ", o.numel());
    };
    check_out(bg_tex_loss, numel_bg, "bg_tex_loss");
    check_out(grids_drift_loss, numel_gc + numel_gf, "grids_drift_loss");
    check_out(grid_camera_tv_loss, numel_gc, "grid_camera_tv_loss");
    check_out(grid_frame_tv_loss, numel_gf, "grid_frame_tv_loss");

    launch_bg_grid_losses_fwd_kernel(
        B_tex,
        D_tex,
        H_tex,
        W_tex,
        C_tex,
        B_gc,
        D_gc,
        H_gc,
        W_gc,
        B_gf,
        D_gf,
        H_gf,
        W_gf,
        static_cast<float>(bg_tex_factor),
        static_cast<float>(grid_drift_camera_factor),
        static_cast<float>(grid_drift_frame_factor),
        static_cast<float>(grid_camera_tv_factor),
        static_cast<float>(grid_frame_tv_factor),
        bg_ptr,
        gc_ptr,
        gf_ptr,
        bg_tex_loss,
        grids_drift_loss,
        grid_camera_tv_loss,
        grid_frame_tv_loss
    );
}

void bg_grid_losses_bwd(
    const at::optional<at::Tensor> &bg_tex,
    int64_t bg_tex_depth,
    const at::optional<at::Tensor> &grids_camera,
    const at::optional<at::Tensor> &grids_frame,
    double bg_tex_factor,
    double grid_drift_camera_factor,
    double grid_drift_frame_factor,
    double grid_camera_tv_factor,
    double grid_frame_tv_factor,
    const at::Tensor &v_bg_tex_loss,
    const at::Tensor &v_grids_drift_loss,
    const at::Tensor &v_grid_camera_tv_loss,
    const at::Tensor &v_grid_frame_tv_loss,
    at::Tensor v_bg_tex,
    at::Tensor v_grids_camera,
    at::Tensor v_grids_frame
)
{
    // Single reference-device guard held through the launch (see fwd).
    const at::Tensor *ref = nullptr;
    if(bg_tex.has_value())
    {
        ref = &bg_tex.value();
    }
    else if(grids_camera.has_value())
    {
        ref = &grids_camera.value();
    }
    else if(grids_frame.has_value())
    {
        ref = &grids_frame.value();
    }
    TORCH_CHECK(
        ref != nullptr,
        "bg_grid_losses_bwd: at least one of bg_tex / grids_camera / "
        "grids_frame must be provided"
    );
    const at::cuda::OptionalCUDAGuard device_guard(device_of(*ref));
    const auto dev = ref->device();

    CHECK_INPUT(v_bg_tex_loss);
    CHECK_INPUT(v_grids_drift_loss);
    CHECK_INPUT(v_grid_camera_tv_loss);
    CHECK_INPUT(v_grid_frame_tv_loss);
    CHECK_INPUT(v_bg_tex);
    CHECK_INPUT(v_grids_camera);
    CHECK_INPUT(v_grids_frame);

    int B_tex = 0, D_tex = 0, H_tex = 0, W_tex = 0, C_tex = 0;
    int B_gc = 0, D_gc = 0, H_gc = 0, W_gc = 0;
    int B_gf = 0, D_gf = 0, H_gf = 0, W_gf = 0;

    const at::Tensor *bg_ptr = nullptr;
    const at::Tensor *gc_ptr = nullptr;
    const at::Tensor *gf_ptr = nullptr;

    if(bg_tex.has_value())
    {
        const auto &t = bg_tex.value();
        CHECK_INPUT(t);
        TORCH_CHECK(t.device() == dev, "bg_tex must be on the same device as the other inputs");
        TORCH_CHECK(t.dim() == 4 && t.scalar_type() == at::kFloat, "bg_tex must be float32 of shape [B*D, H, W, C]");
        const int BD = t.size(0);
        TORCH_CHECK(bg_tex_depth >= 1, "bg_tex_depth must be >= 1 (1 for planar, 6 for cubemap)");
        TORCH_CHECK(BD % bg_tex_depth == 0, "bg_tex.size(0) must be divisible by bg_tex_depth");
        D_tex  = static_cast<int>(bg_tex_depth);
        B_tex  = BD / D_tex;
        H_tex  = t.size(1);
        W_tex  = t.size(2);
        C_tex  = t.size(3);
        bg_ptr = &t;
    }
    if(grids_camera.has_value())
    {
        const auto &t = grids_camera.value();
        CHECK_INPUT(t);
        TORCH_CHECK(t.device() == dev, "grids_camera must be on the same device as the other inputs");
        TORCH_CHECK(
            t.dim() == 4 && t.scalar_type() == at::kFloat,
            "grids_camera must be float32 of shape "
            "[B*12, D, H, W]"
        );
        const int B12 = t.size(0);
        TORCH_CHECK(
            B12 % LOSSES_GRID_NUM_CHANNELS == 0, "grids_camera.size(0) must be divisible by ", LOSSES_GRID_NUM_CHANNELS
        );
        B_gc   = B12 / LOSSES_GRID_NUM_CHANNELS;
        D_gc   = t.size(1);
        H_gc   = t.size(2);
        W_gc   = t.size(3);
        gc_ptr = &t;
    }
    if(grids_frame.has_value())
    {
        const auto &t = grids_frame.value();
        CHECK_INPUT(t);
        TORCH_CHECK(t.device() == dev, "grids_frame must be on the same device as the other inputs");
        TORCH_CHECK(
            t.dim() == 4 && t.scalar_type() == at::kFloat,
            "grids_frame must be float32 of shape "
            "[B*12, D, H, W]"
        );
        const int B12 = t.size(0);
        TORCH_CHECK(
            B12 % LOSSES_GRID_NUM_CHANNELS == 0, "grids_frame.size(0) must be divisible by ", LOSSES_GRID_NUM_CHANNELS
        );
        B_gf   = B12 / LOSSES_GRID_NUM_CHANNELS;
        D_gf   = t.size(1);
        H_gf   = t.size(2);
        W_gf   = t.size(3);
        gf_ptr = &t;
    }

    // Grad buffers must match the loss/grad layout the kernel reads and writes.
    const int64_t numel_bg = static_cast<int64_t>(B_tex) * D_tex * H_tex * W_tex * C_tex;
    const int64_t numel_gc = static_cast<int64_t>(B_gc) * D_gc * H_gc * W_gc;
    const int64_t numel_gf = static_cast<int64_t>(B_gf) * D_gf * H_gf * W_gf;
    auto check_buf         = [&](const at::Tensor &o, int64_t n, const char *name)
    {
        TORCH_CHECK(o.scalar_type() == at::kFloat, name, " must be float32");
        TORCH_CHECK(o.device() == dev, name, " must be on the input device");
        TORCH_CHECK(o.numel() == n, name, " numel mismatch: expected ", n, ", got ", o.numel());
    };
    // Upstream grad buffers (read by computed ranges).
    check_buf(v_bg_tex_loss, numel_bg, "v_bg_tex_loss");
    check_buf(v_grids_drift_loss, numel_gc + numel_gf, "v_grids_drift_loss");
    check_buf(v_grid_camera_tv_loss, numel_gc, "v_grid_camera_tv_loss");
    check_buf(v_grid_frame_tv_loss, numel_gf, "v_grid_frame_tv_loss");
    // Grad-input buffers (written where the matching input is present).
    if(bg_ptr != nullptr)
    {
        check_buf(v_bg_tex, bg_ptr->numel(), "v_bg_tex");
    }
    if(gc_ptr != nullptr)
    {
        check_buf(v_grids_camera, gc_ptr->numel(), "v_grids_camera");
    }
    if(gf_ptr != nullptr)
    {
        check_buf(v_grids_frame, gf_ptr->numel(), "v_grids_frame");
    }

    launch_bg_grid_losses_bwd_kernel(
        B_tex,
        D_tex,
        H_tex,
        W_tex,
        C_tex,
        B_gc,
        D_gc,
        H_gc,
        W_gc,
        B_gf,
        D_gf,
        H_gf,
        W_gf,
        static_cast<float>(bg_tex_factor),
        static_cast<float>(grid_drift_camera_factor),
        static_cast<float>(grid_drift_frame_factor),
        static_cast<float>(grid_camera_tv_factor),
        static_cast<float>(grid_frame_tv_factor),
        bg_ptr,
        gc_ptr,
        gf_ptr,
        v_bg_tex_loss,
        v_grids_drift_loss,
        v_grid_camera_tv_loss,
        v_grid_frame_tv_loss,
        v_bg_tex,
        v_grids_camera,
        v_grids_frame
    );
}

void register_bg_grid_losses_cuda_impl(torch::Library &m)
{
    m.impl("bg_grid_losses_fwd", &bg_grid_losses_fwd);
    m.impl("bg_grid_losses_bwd", &bg_grid_losses_bwd);
}
} // namespace gsplat

#endif // GSPLAT_BUILD_LOSSES
