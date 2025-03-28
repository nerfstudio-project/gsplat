from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor

from gsplat.cuda._wrapper import _make_lazy_cuda_obj


@dataclass
class CameraModelParameters:

    resolution: Tuple[int, int]  # (width, height)
    shutter_type: Literal[
        "GLOBAL",
        "ROLLING_TOP_TO_BOTTOM",
        "ROLLING_LEFT_TO_RIGHT",
        "ROLLING_BOTTOM_TO_TOP",
        "ROLLING_RIGHT_TO_LEFT",
    ]



def compute_max_distance_to_border(image_size_component: float, principal_point_component: float) -> float:
    """Given an image size component (x or y) and corresponding principal point component (x or y),
    returns the maximum distance (in image domain units) from the principal point to either image boundary."""
    center = 0.5 * image_size_component
    if principal_point_component > center:
        return principal_point_component
    else:
        return image_size_component - principal_point_component

def compute_max_radius(image_size: Tuple[int, int], principal_point: Tuple[float, float]) -> float:
    """Compute the maximum radius from the principal point to the image boundaries."""
    max_diag_x = compute_max_distance_to_border(image_size[0], principal_point[0])
    max_diag_y = compute_max_distance_to_border(image_size[1], principal_point[1])
    max_diag = (max_diag_x ** 2 + max_diag_y ** 2) ** 0.5
    return max_diag
