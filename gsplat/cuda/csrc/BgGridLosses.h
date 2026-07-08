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

#pragma once

namespace at
{
class Tensor;
}

namespace gsplat
{
// Fused sky-envmap TV + bilateral-grid drift + bilateral-grid spatial TV.
//
// Inputs are optional: a missing tensor is passed as nullptr. For a present
// tensor, a negative factor disables the corresponding loss and writes its
// expected-shaped zero output; every other factor, including NaN, is applied to
// the per-element contribution. The factor is expected to already include any
// outer mean-reduction `1/N`.
//
// Shapes (after the Python wrapper flattens cubemaps):
//   bg_tex        : [B_tex * D_tex, H_tex, W_tex, C_tex]  (D_tex == 1 planar,
//                                                          D_tex == 6 cubemap)
//   grids_camera  : [B_gc * 12, D_gc, H_gc, W_gc]          (12 = 3x4 affine)
//   grids_frame   : [B_gf * 12, D_gf, H_gf, W_gf]
//
// Output buffers (pre-allocated by the Python caller):
//   bg_tex_loss        : [B_tex * D_tex * H_tex * W_tex * C_tex]
//   grids_drift_loss   : [numel_gc_cells + numel_gf_cells]  where
//                        numel_gX_cells = B_gX * D_gX * H_gX * W_gX (one
//                        scalar drift value per grid cell, NOT per matrix
//                        entry).
//   grid_camera_tv_loss: [numel_gc_cells]
//   grid_frame_tv_loss : [numel_gf_cells]

void launch_bg_grid_losses_fwd_kernel(
    int B_tex,
    int D_tex,
    int H_tex,
    int W_tex,
    int C_tex,
    int B_gc,
    int D_gc,
    int H_gc,
    int W_gc,
    int B_gf,
    int D_gf,
    int H_gf,
    int W_gf,
    float bg_tex_factor,
    float grid_drift_camera_factor,
    float grid_drift_frame_factor,
    float grid_camera_tv_factor,
    float grid_frame_tv_factor,
    const at::Tensor *bg_tex, // may be nullptr / disabled
    const at::Tensor *grids_camera,
    const at::Tensor *grids_frame,
    at::Tensor &bg_tex_loss,
    at::Tensor &grids_drift_loss,
    at::Tensor &grid_camera_tv_loss,
    at::Tensor &grid_frame_tv_loss
);

void launch_bg_grid_losses_bwd_kernel(
    int B_tex,
    int D_tex,
    int H_tex,
    int W_tex,
    int C_tex,
    int B_gc,
    int D_gc,
    int H_gc,
    int W_gc,
    int B_gf,
    int D_gf,
    int H_gf,
    int W_gf,
    float bg_tex_factor,
    float grid_drift_camera_factor,
    float grid_drift_frame_factor,
    float grid_camera_tv_factor,
    float grid_frame_tv_factor,
    const at::Tensor *bg_tex,
    const at::Tensor *grids_camera,
    const at::Tensor *grids_frame,
    const at::Tensor &v_bg_tex_loss,
    const at::Tensor &v_grids_drift_loss,
    const at::Tensor &v_grid_camera_tv_loss,
    const at::Tensor &v_grid_frame_tv_loss,
    at::Tensor &v_bg_tex,
    at::Tensor &v_grids_camera,
    at::Tensor &v_grids_frame
);
} // namespace gsplat
