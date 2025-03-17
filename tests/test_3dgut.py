import math
import os

import imageio
import numpy as np
import torch

from gsplat import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)
from gsplat._helper import load_test_data
from gsplat.cuda._backend import _C
from gsplat.utils import so3_matrix_to_quat

torch.manual_seed(42)

device = torch.device("cuda:0")

def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    colors = colors[None].repeat(len(viewmats), 1, 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }

data = test_data()
Ks = data["Ks"][:1].contiguous()
viewmats = data["viewmats"][:1].contiguous()
height = data["height"]
width = data["width"]
quats = data["quats"].contiguous()
scales = data["scales"].contiguous()
means = data["means"].contiguous()
opacities = data["opacities"].contiguous()
C = len(Ks)
colors = data["colors"][:1].contiguous()

resolution = [width, height]
principal_point = Ks[0, :2, 2].tolist()
focal_length = Ks[0, :2, :2].diag().tolist()
radial_coeffs = [0., 0., 0., 0., 0., 0.]
tangential_coeffs = [0., 0.]
thin_prism_coeffs = [0., 0., 0., 0.]
T_world_sensor_R = viewmats[0, :3, :3].cpu()
T_world_sensor_t = viewmats[0, :3, 3].cpu().numpy()

params = _C.OpenCVPinholeCameraModelParameters()
params.resolution = resolution
params.shutter_type = _C.ShutterType.GLOBAL
params.principal_point = principal_point
params.focal_length = focal_length
params.radial_coeffs = radial_coeffs
params.tangential_coeffs = tangential_coeffs
params.thin_prism_coeffs = thin_prism_coeffs

T_world_sensor_quat = so3_matrix_to_quat(T_world_sensor_R).numpy()[0]
T_world_sensor_tquat = np.hstack([T_world_sensor_t, T_world_sensor_quat])

rs = _C.RollingShutterParameters()
rs.T_world_sensors = np.hstack(
    [T_world_sensor_tquat, T_world_sensor_tquat]
).tolist()  # represents two tquat [t,q] poses at start / end timestamps
rs.timestamps_us = [0, 1]  # arbitrary timestamps

def rasterizer_and_save(radii, means2d, depths, conics, file_name="render.png"):
    # Identify intersecting tiles
    tile_size = 16
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # forward
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities.repeat(C, 1),
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )

    imageio.imsave(
        file_name, (render_colors[0].cpu().numpy() * 255).astype(np.uint8)
    )


radii, means2d, depths, conics, _ = _C.projection_ut_3dgs_fused(
    means.contiguous(),
    quats.contiguous(),
    scales.contiguous(),
    0.3, # eps2d,
    0.01, # near
    1e10, # far
    0.0, # radius_clip
    True, # calc_compensations
    params,
    rs,
    _C.UnscentedTransformParameters()
)
rasterizer_and_save(radii, means2d, depths, conics, "results/ut.png")

radii, means2d, depths, conics, _ = fully_fused_projection(
    means, None, quats, scales, viewmats, Ks, width, height, 0.3, 0.01, 1e10, 0.0
)
rasterizer_and_save(radii, means2d, depths, conics, "results/ewa.png")
