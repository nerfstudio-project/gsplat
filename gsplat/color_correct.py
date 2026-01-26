"""Color correction utilities for post-processing evaluation."""

import torch


def color_correct_quadratic(
    img: torch.Tensor, ref: torch.Tensor, num_iters: int = 5, eps: float = 0.5 / 255
) -> torch.Tensor:
    """
    Warp `img` to match the colors in `ref_img` using iterative color matching.

    This function performs color correction by warping the colors of the input image
    to match those of a reference image. It uses a least squares method to find a
    transformation that maps the input image's colors to the reference image's colors.

    The algorithm iteratively solves a system of linear equations, updating the set of
    unsaturated pixels in each iteration. This approach helps handle non-linear color
    transformations and reduces the impact of clipping.

    Adapted from the implementation at:
    https://github.com/google-research/multinerf

    Args:
        img (torch.Tensor): Input image to be color corrected. Shape: [..., num_channels]
        ref (torch.Tensor): Reference image to match colors. Shape: [..., num_channels]
        num_iters (int, optional): Number of iterations for the color matching process.
                                   Default is 5.
        eps (float, optional): Small value to determine the range of unclipped pixels.
                               Default is 0.5 / 255.

    Returns:
        torch.Tensor: Color corrected image with the same shape as the input image.

    Note:
        - Both input and reference images should be in the range [0, 1].
        - The function works with any number of channels, but typically used with 3 (RGB).
    """
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f"img's {img.shape[-1]} and ref's {ref.shape[-1]} channels must match"
        )
    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])
    ref_mat = ref.reshape([-1, num_channels])

    def is_unclipped(z):
        return (z >= eps) & (z <= 1 - eps)  # z \in [eps, 1-eps].

    mask0 = is_unclipped(img_mat)
    # Because the set of saturated pixels may change after solving for a
    # transformation, we repeatedly solve a system `num_iters` times and update
    # our estimate of which pixels are saturated.
    for _ in range(num_iters):
        # Construct the left hand side of a linear system that contains a quadratic
        # expansion of each pixel of `img`.
        a_mat = []
        for c in range(num_channels):
            # Quadratic term.
            a_mat.append(img_mat[:, c : (c + 1)] * img_mat[:, c:])
        a_mat.append(img_mat)  # Linear term.
        a_mat.append(torch.ones_like(img_mat[:, :1]))  # Bias term.
        a_mat = torch.cat(a_mat, dim=-1)
        warp = []
        for c in range(num_channels):
            # Construct the right hand side of a linear system containing each color
            # of `ref`.
            b = ref_mat[:, c]
            # Ignore rows of the linear system that were saturated in the input or are
            # saturated in the current corrected color estimate.
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = torch.where(mask[:, None], a_mat, torch.zeros_like(a_mat))
            mb = torch.where(mask, b, torch.zeros_like(b))
            w = torch.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
            assert torch.all(torch.isfinite(w))
            warp.append(w)
        warp = torch.stack(warp, dim=-1)
        # Apply the warp to update img_mat.
        img_mat = torch.clip(torch.matmul(a_mat, warp), 0, 1)
    corrected_img = torch.reshape(img_mat, img.shape)
    return corrected_img


def color_correct_affine(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Warp `img` to match the colors in `ref` using per-channel affine transformation.

    This function computes a per-channel affine best fit (a * ref + b = img) from the reference
    to the input image, then applies the inverse mapping to correct the input image
    back to match the reference.

    Adapted from the implementation at:
    https://github.com/google-research/multinerf

    Args:
        img (torch.Tensor): Input image to be color corrected. Shape: [..., num_channels]
        ref (torch.Tensor): Reference image to match colors. Shape: [..., num_channels]

    Returns:
        torch.Tensor: Color corrected image with the same shape as the input image.

    Note:
        - Both input and reference images should be in the range [0, 1].
        - The function works with any number of channels, but typically used with 3 (RGB).
    """
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f"img's {img.shape[-1]} and ref's {ref.shape[-1]} channels must match"
        )
    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])  # [N, C]
    ref_mat = ref.reshape([-1, num_channels])  # [N, C]

    # Compute per-channel means
    ref_mean = ref_mat.mean(dim=0)  # [C]
    img_mean = img_mat.mean(dim=0)  # [C]
    ref_img_mean = (ref_mat * img_mat).mean(dim=0)  # [C]
    ref_ref_mean = (ref_mat * ref_mat).mean(dim=0)  # [C]

    # Best fit affine: a * ref + b = img (mapping ref -> img)
    # slope a = Cov(ref, img) / Var(ref)
    var_ref = ref_ref_mean - ref_mean * ref_mean
    # Clamp variance away from zero to avoid division by zero / NaN
    var_ref = torch.clamp(var_ref, min=1e-8)
    a = (ref_img_mean - ref_mean * img_mean) / var_ref
    b = img_mean - a * ref_mean

    # Inverse mapping: corrected = (img - b) / a to map img back to match ref
    # Clamp 'a' away from zero to avoid NaN
    a = torch.where(a.abs() < 1e-8, torch.ones_like(a), a)
    corrected_mat = (img_mat - b) / a
    corrected_mat = torch.clip(corrected_mat, 0, 1)

    corrected_img = torch.reshape(corrected_mat, img.shape)
    return corrected_img
