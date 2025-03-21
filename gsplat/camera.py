from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import Tensor

from gsplat.cuda._wrapper import _make_lazy_cuda_obj


@dataclass
class CameraModelParameters:

    resolution: Tuple[int, int] # (width, height)
    shutter_type: Literal[
        "GLOBAL",
        "ROLLING_TOP_TO_BOTTOM",
        "ROLLING_LEFT_TO_RIGHT",
        "ROLLING_BOTTOM_TO_TOP",
        "ROLLING_RIGHT_TO_LEFT"
    ]

    
@dataclass
class OpenCVPinholeCameraModelParameters(CameraModelParameters):

    principal_point: Tuple[float, float]
    focal_length: Tuple[float, float]
    radial_coeffs: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    tangential_coeffs: Tuple[float, float] = (0.0, 0.0)
    thin_prism_coeffs: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def to_cpp(self):
        p = _make_lazy_cuda_obj("OpenCVPinholeCameraModelParameters")()
        p.resolution = self.resolution
        p.shutter_type = _make_lazy_cuda_obj(f"ShutterType.{self.shutter_type}")
        p.principal_point = self.principal_point
        p.focal_length = self.focal_length
        p.radial_coeffs = self.radial_coeffs
        p.tangential_coeffs = self.tangential_coeffs
        p.thin_prism_coeffs = self.thin_prism_coeffs
        return p

@dataclass
class OpenCVFisheyeCameraModelParameters(CameraModelParameters):
    
    principal_point: Tuple[float, float]
    focal_length: Tuple[float, float]
    radial_coeffs: Tuple[float, float, float, float]

    def to_cpp(self):
        p = _make_lazy_cuda_obj("OpenCVFisheyeCameraModelParameters")()
        p.resolution = self.resolution
        p.shutter_type = _make_lazy_cuda_obj(f"ShutterType.{self.shutter_type}")
        p.principal_point = self.principal_point
        p.focal_length = self.focal_length
        p.radial_coeffs = self.radial_coeffs
        return p

@dataclass
class RollingShutterParameters:

    T_world_sensors: Tuple[float, ...] # 7 * 2
    timestamps_us: Tuple[int, int]

    def to_cpp(self):
        p = _make_lazy_cuda_obj("RollingShutterParameters")()
        p.T_world_sensors = self.T_world_sensors
        p.timestamps_us = self.timestamps_us
        return p


def to_params(
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
):
    import numpy as np

    from gsplat.utils import so3_matrix_to_quat

    C = viewmats.size(0)
    assert C == 1, "Only support single camera for now"
    
    # check the R part in the viewmats are orthonormal
    R = viewmats[:, :3, :3]
    det = torch.det(R)
    assert torch.allclose(det, torch.ones_like(det)), "The R part in the viewmats should be orthonormal"

    if camera_model == "pinhole":
        cm_params = OpenCVPinholeCameraModelParameters(
            resolution=(width, height),
            shutter_type="GLOBAL",
            principal_point=Ks[0, :2, 2].tolist(),
            focal_length=Ks[0, :2, :2].diag().tolist(),
            radial_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            tangential_coeffs=(0.0, 0.0),
            thin_prism_coeffs=(0.0, 0.0, 0.0, 0.0)
        )
    else:
        raise NotImplementedError(f"Camera model {camera_model} is not supported")

    T_world_sensor_R = viewmats[0, :3, :3].cpu()
    T_world_sensor_quat = so3_matrix_to_quat(T_world_sensor_R).numpy()[0]
    T_world_sensor_t = viewmats[0, :3, 3].cpu().numpy()
    T_world_sensor_tquat = np.hstack([T_world_sensor_t, T_world_sensor_quat])

    rs_params = RollingShutterParameters(
        T_world_sensors = np.hstack(
            [T_world_sensor_tquat, T_world_sensor_tquat]
        ).tolist(),  # represents two tquat [t,q] poses at start / end timestamps
        timestamps_us = [0, 1]  # arbitrary timestamps
    )
    return cm_params, rs_params
