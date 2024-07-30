import torch
from torch import Tensor


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    """Convert normalized quaternion to rotation matrix.

    Args:
        quat: Normalized quaternion in wxyz convension. (..., 4)

    Returns:
        Rotation matrix (..., 3, 3)
    """
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def sh_to_rgb(sh: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def log_transform(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def inverse_log_transform(y):
    return torch.sign(y) * (torch.expm1(torch.abs(y)))
