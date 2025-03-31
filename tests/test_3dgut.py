import math
import os

import imageio
import numpy as np
import torch
from torch import Tensor

from gsplat import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterization,
    rasterize_to_pixels,
)
from gsplat._helper import load_test_data
from gsplat.cuda._wrapper import (
    RollingShutterType,
    fully_fused_projection_with_ut,
    rasterize_to_pixels_eval3d,
)

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
viewmats_rs = viewmats.clone()


# render_colors_undistorted, _, _ = rasterization(
#     means,
#     quats,
#     scales,
#     opacities, 
#     colors,
#     viewmats,
#     Ks,
#     width,
#     height,
#     packed=False,
#     camera_model="pinhole",
#     with_ut=True,
#     with_eval3d=True,
# )


def opencv_lens_distortion(
    uv: Tensor,
    radial_coeffs: Tensor = None,
    tangential_coeffs: Tensor = None,
    thin_prism_coeffs: Tensor = None,
) -> Tensor:
    """The opencv camera distortion of {k1, k2, p1, p2, k3, k4, k5, k6}.

    See https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html for more details.
    """
    if radial_coeffs is None:
        k1, k2, k3, k4, k5, k6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        k1, k2, k3, k4, k5, k6 = radial_coeffs.unbind(dim=-1)
    if tangential_coeffs is None:
        p1, p2 = 0.0, 0.0
    else:
        p1, p2 = tangential_coeffs.unbind(dim=-1)
    if thin_prism_coeffs is None:
        s1, s2, s3, s4 = 0.0, 0.0, 0.0, 0.0
    else:
        s1, s2, s3, s4 = thin_prism_coeffs.unbind(dim=-1)

    assert uv.ndim == 2

    u, v = torch.unbind(uv, dim=-1)
    r2 = u * u + v * v
    r4 = r2**2
    r6 = r4 * r2
    ratial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (
        1 + k4 * r2 + k5 * r4 + k6 * r6
    )
    if ratial < 0:
        print ("Warning: negative radial distortion factor")
    fx = 2 * p1 * u * v + p2 * (r2 + 2 * u * u) + s1 * r2 + s2 * r4
    fy = 2 * p2 * u * v + p1 * (r2 + 2 * v * v) + s3 * r2 + s4 * r4
    return torch.stack([u * ratial + fx, v * ratial + fy], dim=-1)

# converged: 1, iter: 7, xd: -0.654764, yd: -0.433112, x: 1.133525, y: 0.749801
uv = torch.tensor([[1.133525, 0.749801]])
radial_coeffs = torch.tensor([-0.3, -0.3, 0.0, 0.0, 0.0, 0.0])  # k1, k2, k3, k4, k5, k6
print(opencv_lens_distortion(uv, radial_coeffs))


# import cv2

# K = Ks[0].cpu().numpy()
# fx = K[0, 0]
# fy = K[1, 1]
# cx = K[0, 2]
# cy = K[1, 2]

# x, y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")
# x = x.flatten()
# y = y.flatten()
# u = (x - cx) / fx
# v = (y - cy) / fy
# uv = torch.stack([u, v], dim=-1)  # (H*W, 2)
# uv_prime = opencv_lens_distortion(
#     uv,
#     radial_coeffs=torch.tensor([2.3, 2.3, 0.0, 0.0, 0.0, 0.0]),  # k1, k2, k3, k4, k5, k6
#     tangential_coeffs=torch.tensor([0.0, 0.0]),  # p1, p2
#     thin_prism_coeffs=torch.tensor([0.0, 0.0, 0.0, 0.0]),  # s1, s2, s3, s4
# )
# x_prime = uv_prime[:, 0] * fx + cx
# y_prime = uv_prime[:, 1] * fy + cy
# mapx = x_prime.reshape(height, width).numpy()
# mapy = y_prime.reshape(height, width).numpy()

# # params = np.array([2.3, 2.3, 0.0, 0.0], dtype=np.float32)  # k1, k2, p1, p2
# # mapx, mapy = cv2.initUndistortRectifyMap(
# #     K, params, None, K, (width, height), cv2.CV_32FC1
# # )
# image = (render_colors_undistorted[0].cpu().numpy() * 255).astype(np.uint8)  # (H, W, C)
# image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
# imageio.imsave("results/render.jpg", image)

# print ("uv_normalized", uv[:5])
# print ("uvND", uv_prime[:5])



id = 86704

# scales[id:id+1, -1] *=4
render_colors, render_alphas, info = rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width,
    height,
    packed=False,
    camera_model="fisheye",
    with_ut=True,
    with_eval3d=True,
    radial_coeffs=torch.tensor([[-.5, -.5, 0., 0.]], device=device),
    tangential_coeffs=None,
    thin_prism_coeffs=None,
    rolling_shutter=RollingShutterType.GLOBAL,
    viewmats_rs=None,
)
radii = info["radii"]

print (torch.where(radii == radii.max()))

canvas = np.vstack([
    (render_colors[0].cpu().numpy() * 255).astype(np.uint8),
])
print (canvas.shape)

imageio.imsave("results/render.jpg", canvas)
