import math
import struct
from io import BytesIO
from typing import Literal, Optional

import numpy as np
import torch


def sh2rgb(sh: torch.Tensor) -> torch.Tensor:
    """Convert Sphere Harmonics to RGB

    Args:
        sh (torch.Tensor): SH tensor

    Returns:
        torch.Tensor: RGB tensor
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def part1by2_vec(x: torch.Tensor) -> torch.Tensor:
    """Interleave bits of x with 0s

    Args:
        x (torch.Tensor): Input tensor. Shape (N,)

    Returns:
        torch.Tensor: Output tensor. Shape (N,)
    """

    x = x & 0x000003FF
    x = (x ^ (x << 16)) & 0xFF0000FF
    x = (x ^ (x << 8)) & 0x0300F00F
    x = (x ^ (x << 4)) & 0x030C30C3
    x = (x ^ (x << 2)) & 0x09249249
    return x


def encode_morton3_vec(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """Compute Morton codes for 3D coordinates

    Args:
        x (torch.Tensor): X coordinates. Shape (N,)
        y (torch.Tensor): Y coordinates. Shape (N,)
        z (torch.Tensor): Z coordinates. Shape (N,)
    Returns:
        torch.Tensor: Morton codes. Shape (N,)
    """
    return (part1by2_vec(z) << 2) + (part1by2_vec(y) << 1) + part1by2_vec(x)


def sort_centers(centers: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Sort centers based on Morton codes

    Args:
        centers (torch.Tensor): Centers. Shape (N, 3)
        indices (torch.Tensor): Indices. Shape (N,)
    Returns:
        torch.Tensor: Sorted indices. Shape (N,)
    """
    # Compute min and max values in a single operation
    min_vals, _ = torch.min(centers, dim=0)
    max_vals, _ = torch.max(centers, dim=0)

    # Compute the scaling factors
    lengths = max_vals - min_vals
    lengths[lengths == 0] = 1  # Prevent division by zero

    # Normalize and scale to 10-bit integer range (0-1024)
    scaled_centers = ((centers - min_vals) / lengths * 1024).floor().to(torch.int32)

    # Extract x, y, z coordinates
    x, y, z = scaled_centers[:, 0], scaled_centers[:, 1], scaled_centers[:, 2]

    # Compute Morton codes using vectorized operations
    morton = encode_morton3_vec(x, y, z)

    # Sort indices based on Morton codes
    sorted_indices = indices[torch.argsort(morton).to(indices.device)]

    return sorted_indices


def pack_unorm(value: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack a floating point value into an unsigned integer with a given number of bits.

    Args:
        value (torch.Tensor): Floating point value to pack. Shape (N,)
        bits (int): Number of bits to pack into.

    Returns:
        torch.Tensor: Packed value. Shape (N,)
    """

    t = (1 << bits) - 1
    packed = torch.clamp((value * t + 0.5).floor(), min=0, max=t)
    # Convert to integer type
    return packed.to(torch.int64)


def pack_111011(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Pack three floating point values into a 32-bit integer with 11, 10, and 11 bits.

    Args:
        x (torch.Tensor): X component. Shape (N,)
        y (torch.Tensor): Y component. Shape (N,)
        z (torch.Tensor): Z component. Shape (N,)
    Returns:
        torch.Tensor: Packed values. Shape (N,)
    """
    # Pack each component using pack_unorm
    packed_x = pack_unorm(x, 11) << 21
    packed_y = pack_unorm(y, 10) << 11
    packed_z = pack_unorm(z, 11)

    # Combine the packed values using bitwise OR
    return packed_x | packed_y | packed_z


def pack_8888(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """Pack four floating point values into a 32-bit integer with 8 bits each.

    Args:
        x (torch.Tensor): X component. Shape (N,)
        y (torch.Tensor): Y component. Shape (N,)
        z (torch.Tensor): Z component. Shape (N,)
        w (torch.Tensor): W component. Shape (N,)
    Returns:
        torch.Tensor: Packed values. Shape (N,)
    """
    # Pack each component using pack_unorm
    packed_x = pack_unorm(x, 8) << 24
    packed_y = pack_unorm(y, 8) << 16
    packed_z = pack_unorm(z, 8) << 8
    packed_w = pack_unorm(w, 8)

    # Combine the packed values using bitwise OR
    return packed_x | packed_y | packed_z | packed_w


def pack_rotation(q: torch.Tensor) -> torch.Tensor:
    """Pack a quaternion into a 32-bit integer.

    Args:
        q (torch.Tensor): Quaternions. Shape (N, 4)

    Returns:
        torch.Tensor: Packed values. Shape (N,)
    """

    # Normalize each quaternion
    norms = torch.linalg.norm(q, dim=-1, keepdim=True)
    q = q / norms

    # Find the largest component index for each quaternion
    largest_components = torch.argmax(torch.abs(q), dim=-1)

    # Flip quaternions where the largest component is negative
    batch_indices = torch.arange(q.size(0), device=q.device)
    largest_values = q[batch_indices, largest_components]
    flip_mask = largest_values < 0
    q[flip_mask] *= -1

    # Precomputed indices for the components to pack (excluding largest)
    precomputed_indices = torch.tensor(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=torch.long, device=q.device
    )

    # Gather components to pack for each quaternion
    pack_indices = precomputed_indices[largest_components]
    components_to_pack = q[batch_indices[:, None], pack_indices]

    # Scale and pack each component into 10-bit integers
    norm = math.sqrt(2) * 0.5
    scaled = components_to_pack * norm + 0.5
    packed = pack_unorm(scaled, 10)  # Assuming pack_unorm is vectorized

    # Combine into the final 32-bit integer
    largest_packed = largest_components.to(torch.int64) << 30
    c0_packed = packed[:, 0] << 20
    c1_packed = packed[:, 1] << 10
    c2_packed = packed[:, 2]

    result = largest_packed | c0_packed | c1_packed | c2_packed
    return result


def splat2ply_bytes_compressed(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    chunk_max_size: int = 256,
    opacity_threshold: float = 1 / 255,
) -> bytes:
    """Return the binary compressed Ply file. Used by Supersplat viewer.

    Args:
        means (torch.Tensor): Splat means. Shape (N, 3)
        scales (torch.Tensor): Splat scales. Shape (N, 3)
        quats (torch.Tensor): Splat quaternions. Shape (N, 4)
        opacities (torch.Tensor): Splat opacities. Shape (N,)
        sh0 (torch.Tensor): Spherical harmonics. Shape (N, 3)
        shN (torch.Tensor): Spherical harmonics. Shape (N, K*3)
        chunk_max_size (int): Maximum number of splats per chunk. Default: 256
        opacity_threshold (float): Opacity threshold. Default: 1 / 255

    Returns:
        bytes: Binary compressed Ply file representing the model.
    """

    # Filter the splats with too low opacity
    mask = torch.sigmoid(opacities) > opacity_threshold
    means = means[mask]
    scales = scales[mask]
    sh0_colors = sh2rgb(sh0)
    sh0_colors = sh0_colors[mask]
    shN = shN[mask]
    quats = quats[mask]
    opacities = opacities[mask]

    num_splats = means.shape[0]
    n_chunks = num_splats // chunk_max_size + (num_splats % chunk_max_size != 0)
    indices = torch.arange(num_splats)
    indices = sort_centers(means, indices)

    float_properties = [
        "min_x",
        "min_y",
        "min_z",
        "max_x",
        "max_y",
        "max_z",
        "min_scale_x",
        "min_scale_y",
        "min_scale_z",
        "max_scale_x",
        "max_scale_y",
        "max_scale_z",
        "min_r",
        "min_g",
        "min_b",
        "max_r",
        "max_g",
        "max_b",
    ]
    uint_properties = [
        "packed_position",
        "packed_rotation",
        "packed_scale",
        "packed_color",
    ]
    buffer = BytesIO()

    # Write PLY header
    buffer.write(b"ply\n")
    buffer.write(b"format binary_little_endian 1.0\n")
    buffer.write(f"element chunk {n_chunks}\n".encode())
    for prop in float_properties:
        buffer.write(f"property float {prop}\n".encode())
    buffer.write(f"element vertex {num_splats}\n".encode())
    for prop in uint_properties:
        buffer.write(f"property uint {prop}\n".encode())
    buffer.write(f"element sh {num_splats}\n".encode())
    for j in range(shN.shape[1]):
        buffer.write(f"property uchar f_rest_{j}\n".encode())
    buffer.write(b"end_header\n")

    chunk_data = []
    splat_data = []
    sh_data = []
    for chunk_idx in range(n_chunks):
        chunk_end_idx = min((chunk_idx + 1) * chunk_max_size, num_splats)
        chunk_start_idx = chunk_idx * chunk_max_size
        splat_idxs = indices[chunk_start_idx:chunk_end_idx]

        # Bounds
        # Means
        chunk_means = means[splat_idxs]
        min_means = torch.min(chunk_means, dim=0).values
        max_means = torch.max(chunk_means, dim=0).values
        mean_bounds = torch.cat([min_means, max_means])
        # Scales
        chunk_scales = scales[splat_idxs]
        min_scales = torch.min(chunk_scales, dim=0).values
        max_scales = torch.max(chunk_scales, dim=0).values
        min_scales = torch.clamp(min_scales, -20, 20)
        max_scales = torch.clamp(max_scales, -20, 20)
        scale_bounds = torch.cat([min_scales, max_scales])
        # Colors
        chunk_colors = sh0_colors[splat_idxs]
        min_colors = torch.min(chunk_colors, dim=0).values
        max_colors = torch.max(chunk_colors, dim=0).values
        color_bounds = torch.cat([min_colors, max_colors])
        chunk_data.extend([mean_bounds, scale_bounds, color_bounds])

        # Quantized properties:
        # Means
        normalized_means = (chunk_means - min_means) / (max_means - min_means)
        means_i = pack_111011(
            normalized_means[:, 0],
            normalized_means[:, 1],
            normalized_means[:, 2],
        )
        # Quaternions
        chunk_quats = quats[splat_idxs]
        quat_i = pack_rotation(chunk_quats)
        # Scales
        normalized_scales = (chunk_scales - min_scales) / (max_scales - min_scales)
        scales_i = pack_111011(
            normalized_scales[:, 0],
            normalized_scales[:, 1],
            normalized_scales[:, 2],
        )
        # Colors
        normalized_colors = (chunk_colors - min_colors) / (max_colors - min_colors)
        chunk_opacities = opacities[splat_idxs]
        chunk_opacities = 1 / (1 + torch.exp(-chunk_opacities))
        chunk_opacities = chunk_opacities.unsqueeze(-1)
        normalized_colors_i = torch.cat([normalized_colors, chunk_opacities], dim=-1)
        color_i = pack_8888(
            normalized_colors_i[:, 0],
            normalized_colors_i[:, 1],
            normalized_colors_i[:, 2],
            normalized_colors_i[:, 3],
        )
        splat_data_chunk = torch.stack([means_i, quat_i, scales_i, color_i], dim=1)
        splat_data_chunk = splat_data_chunk.ravel().to(torch.int64)
        splat_data.extend([splat_data_chunk])

        # Quantized spherical harmonics
        shN_chunk = shN[splat_idxs]
        shN_chunk_quantized = (shN_chunk / 8 + 0.5) * 256
        shN_chunk_quantized = torch.clamp(torch.trunc(shN_chunk_quantized), 0, 255)
        shN_chunk_quantized = shN_chunk_quantized.to(torch.uint8)
        sh_data.extend([shN_chunk_quantized.ravel()])

    float_dtype = np.dtype(np.float32).newbyteorder("<")
    uint32_dtype = np.dtype(np.uint32).newbyteorder("<")
    uint8_dtype = np.dtype(np.uint8)

    buffer.write(
        torch.cat(chunk_data).detach().cpu().numpy().astype(float_dtype).tobytes()
    )
    buffer.write(
        torch.cat(splat_data).detach().cpu().numpy().astype(uint32_dtype).tobytes()
    )
    buffer.write(
        torch.cat(sh_data).detach().cpu().numpy().astype(uint8_dtype).tobytes()
    )

    return buffer.getvalue()


def splat2ply_bytes(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
) -> bytes:
    """Return the binary Ply file. Supported by almost all viewers.

    Args:
        means (torch.Tensor): Splat means. Shape (N, 3)
        scales (torch.Tensor): Splat scales. Shape (N, 3)
        quats (torch.Tensor): Splat quaternions. Shape (N, 4)
        opacities (torch.Tensor): Splat opacities. Shape (N,)
        sh0 (torch.Tensor): Spherical harmonics. Shape (N, 3)
        shN (torch.Tensor): Spherical harmonics. Shape (N, K*3)

    Returns:
        bytes: Binary Ply file representing the model.
    """

    num_splats = means.shape[0]
    buffer = BytesIO()

    # Write PLY header
    buffer.write(b"ply\n")
    buffer.write(b"format binary_little_endian 1.0\n")
    buffer.write(f"element vertex {num_splats}\n".encode())
    buffer.write(b"property float x\n")
    buffer.write(b"property float y\n")
    buffer.write(b"property float z\n")
    for i, data in enumerate([sh0, shN]):
        prefix = "f_dc" if i == 0 else "f_rest"
        for j in range(data.shape[1]):
            buffer.write(f"property float {prefix}_{j}\n".encode())
    buffer.write(b"property float opacity\n")
    for i in range(scales.shape[1]):
        buffer.write(f"property float scale_{i}\n".encode())
    for i in range(quats.shape[1]):
        buffer.write(f"property float rot_{i}\n".encode())
    buffer.write(b"end_header\n")

    # Concatenate all tensors in the correct order
    splat_data = torch.cat(
        [means, sh0, shN, opacities.unsqueeze(1), scales, quats], dim=1
    )
    # Ensure correct dtype
    splat_data = splat_data.to(torch.float32)

    # Write binary data
    float_dtype = np.dtype(np.float32).newbyteorder("<")
    buffer.write(splat_data.detach().cpu().numpy().astype(float_dtype).tobytes())

    return buffer.getvalue()


def splat2splat_bytes(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
) -> bytes:
    """Return the binary Splat file. Supported by antimatter15 viewer.

    Args:
        means (torch.Tensor): Splat means. Shape (N, 3)
        scales (torch.Tensor): Splat scales. Shape (N, 3)
        quats (torch.Tensor): Splat quaternions. Shape (N, 4)
        opacities (torch.Tensor): Splat opacities. Shape (N,)
        sh0 (torch.Tensor): Spherical harmonics. Shape (N, 3)

    Returns:
        bytes: Binary Splat file representing the model.
    """

    # Preprocess
    scales = torch.exp(scales)
    sh0_color = sh2rgb(sh0)
    colors = torch.cat([sh0_color, torch.sigmoid(opacities).unsqueeze(-1)], dim=1)
    colors = (colors * 255).clamp(0, 255).to(torch.uint8)

    rots = (quats / torch.linalg.norm(quats, dim=1, keepdim=True)) * 128 + 128
    rots = rots.clamp(0, 255).to(torch.uint8)

    # Sort splats
    num_splats = means.shape[0]
    indices = sort_centers(means, torch.arange(num_splats))

    # Reorder everything
    means = means[indices]
    scales = scales[indices]
    colors = colors[indices]
    rots = rots[indices]

    float_dtype = np.dtype(np.float32).newbyteorder("<")
    means_np = means.detach().cpu().numpy().astype(float_dtype)
    scales_np = scales.detach().cpu().numpy().astype(float_dtype)
    colors_np = colors.detach().cpu().numpy().astype(np.uint8)
    rots_np = rots.detach().cpu().numpy().astype(np.uint8)

    buffer = BytesIO()
    for i in range(num_splats):
        buffer.write(means_np[i].tobytes())
        buffer.write(scales_np[i].tobytes())
        buffer.write(colors_np[i].tobytes())
        buffer.write(rots_np[i].tobytes())

    return buffer.getvalue()


def export_splats(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    format: Literal["ply", "splat", "ply_compressed"] = "ply",
    save_to: Optional[str] = None,
) -> bytes:
    """Export a Gaussian Splats model to bytes.
    The three supported formats are:
    - ply: A standard PLY file format. Supported by most viewers.
    - splat: A custom Splat file format. Supported by antimatter15 viewer.
    - ply_compressed: A compressed PLY file format. Used by Supersplat viewer.

    Args:
        means (torch.Tensor): Splat means. Shape (N, 3)
        scales (torch.Tensor): Splat scales. Shape (N, 3)
        quats (torch.Tensor): Splat quaternions. Shape (N, 4)
        opacities (torch.Tensor): Splat opacities. Shape (N,)
        sh0 (torch.Tensor): Spherical harmonics. Shape (N, 1, 3)
        shN (torch.Tensor): Spherical harmonics. Shape (N, K, 3)
        format (str): Export format. Options: "ply", "splat", "ply_compressed". Default: "ply"
        save_to (str): Output file path. If provided, the bytes will be written to file.
    """
    total_splats = means.shape[0]
    assert means.shape == (total_splats, 3), "Means must be of shape (N, 3)"
    assert scales.shape == (total_splats, 3), "Scales must be of shape (N, 3)"
    assert quats.shape == (total_splats, 4), "Quaternions must be of shape (N, 4)"
    assert opacities.shape == (total_splats,), "Opacities must be of shape (N,)"
    assert sh0.shape == (total_splats, 1, 3), "sh0 must be of shape (N, 1, 3)"
    assert (
        shN.ndim == 3 and shN.shape[0] == total_splats and shN.shape[2] == 3
    ), f"shN must be of shape (N, K, 3), got {shN.shape}"

    # Reshape spherical harmonics
    sh0 = sh0.squeeze(1)  # Shape (N, 3)
    shN = shN.permute(0, 2, 1).reshape(means.shape[0], -1)  # Shape (N, K * 3)

    # Check for NaN or Inf values
    invalid_mask = (
        torch.isnan(means).any(dim=1)
        | torch.isinf(means).any(dim=1)
        | torch.isnan(scales).any(dim=1)
        | torch.isinf(scales).any(dim=1)
        | torch.isnan(quats).any(dim=1)
        | torch.isinf(quats).any(dim=1)
        | torch.isnan(opacities).any(dim=0)
        | torch.isinf(opacities).any(dim=0)
        | torch.isnan(sh0).any(dim=1)
        | torch.isinf(sh0).any(dim=1)
        | torch.isnan(shN).any(dim=1)
        | torch.isinf(shN).any(dim=1)
    )

    # Filter out invalid entries
    valid_mask = ~invalid_mask
    means = means[valid_mask]
    scales = scales[valid_mask]
    quats = quats[valid_mask]
    opacities = opacities[valid_mask]
    sh0 = sh0[valid_mask]
    shN = shN[valid_mask]

    if format == "ply":
        data = splat2ply_bytes(means, scales, quats, opacities, sh0, shN)
    elif format == "splat":
        data = splat2splat_bytes(means, scales, quats, opacities, sh0)
    elif format == "ply_compressed":
        data = splat2ply_bytes_compressed(means, scales, quats, opacities, sh0, shN)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if save_to:
        with open(save_to, "wb") as binary_file:
            binary_file.write(data)

    return data
