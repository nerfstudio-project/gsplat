import open3d as o3d
import numpy as np
import torch
from typing import List, Dict
from pathlib import Path


def load_ply(filepath: str) -> Dict[str, torch.Tensor]:
    """
    Load a PLY file and extract all its attributes into a dictionary of torch tensors.
    """
    print(f"Loading PLY from {filepath}")
    pcd = o3d.t.io.read_point_cloud(filepath)

    # Initialize dictionary for all attributes
    data = {}

    # Extract point positions (means)
    means = pcd.point.positions.numpy()
    data["means"] = torch.from_numpy(means)

    # Extract scales
    scales = np.column_stack([
        pcd.point[f"scale_{i}"].numpy() for i in range(3)
    ])
    data["scales"] = torch.from_numpy(scales)

    # Extract quaternions
    quats = np.column_stack([
        pcd.point[f"rot_{i}"].numpy() for i in range(4)
    ])
    data["quats"] = torch.from_numpy(quats)

    # Extract opacities
    data["opacities"] = torch.from_numpy(pcd.point["opacity"].numpy())

    # Check if we have SH coefficients or colors
    if "f_dc_0" in pcd.point:
        # Count number of SH coefficients
        dc_count = sum(1 for key in pcd.point if key.startswith("f_dc_"))
        rest_count = sum(1 for key in pcd.point if key.startswith("f_rest_"))

        if rest_count > 0:  # We have SH coefficients
            # Extract SH0 coefficients
            sh0 = np.column_stack([
                pcd.point[f"f_dc_{i}"].numpy() for i in range(dc_count)
            ])
            data["sh0"] = torch.from_numpy(sh0).reshape(-1, dc_count // 3, 3).transpose(1, 2)

            # Extract SHN coefficients
            shN = np.column_stack([
                pcd.point[f"f_rest_{i}"].numpy() for i in range(rest_count)
            ])
            data["shN"] = torch.from_numpy(shN).reshape(-1, rest_count // 3, 3).transpose(1, 2)
        else:  # We have colors
            colors = np.column_stack([
                pcd.point[f"f_dc_{i}"].numpy() for i in range(dc_count)
            ])
            data["colors"] = torch.from_numpy(colors * 0.2820947917738781 + 0.5)

    return data


def merge_plys(filepaths: List[str]) -> Dict[str, torch.Tensor]:
    """
    Merge multiple PLY files into a single dictionary of torch tensors.
    """
    print(f"Merging {len(filepaths)} PLY files")
    merged_data = {}

    # Load and merge each PLY file
    for filepath in filepaths:
        data = load_ply(filepath)

        # For first file, initialize merged_data
        if not merged_data:
            merged_data = {k: [v] for k, v in data.items()}
        else:
            # Verify that the current file has the same attributes
            assert set(data.keys()) == set(merged_data.keys()), \
                f"PLY file {filepath} has different attributes than previous files"

            # Append tensors to lists
            for k, v in data.items():
                merged_data[k].append(v)

    # Concatenate all tensors
    return {k: torch.cat(v, dim=0) for k, v in merged_data.items()}


def save_merged_ply(merged_data: Dict[str, torch.Tensor], output_path: str):
    """
    Save merged data as a PLY file using the same format as the original save_ply function.
    """
    # Convert to ParameterDict if it isn't already
    if not isinstance(merged_data, torch.nn.ParameterDict):
        param_dict = torch.nn.ParameterDict()
        for k, v in merged_data.items():
            if k != "colors":  # Skip colors if present
                param_dict[k] = torch.nn.Parameter(v)
        merged_data = param_dict

    # Use the provided save_ply function
    colors = merged_data.get("colors") if "colors" in merged_data else None
    save_ply(merged_data, output_path, colors)


def process_plys(input_dir: str, output_path: str, pattern: str = "*.ply"):
    """
    Process all PLY files in a directory and merge them into a single file.

    Args:
        input_dir: Directory containing input PLY files
        output_path: Path where to save the merged PLY file
        pattern: Glob pattern to match PLY files (default: "*.ply")
    """
    input_path = Path(input_dir)
    ply_files = sorted(str(p) for p in input_path.glob(pattern))

    if not ply_files:
        raise ValueError(f"No PLY files found in {input_dir} matching pattern {pattern}")

    print(f"Found {len(ply_files)} PLY files")

    # Merge all PLY files
    merged_data = merge_plys(ply_files)

    # Save merged result
    save_merged_ply(merged_data, output_path)
    print(f"Saved merged PLY to {output_path}")

def save_ply(splats: torch.nn.ParameterDict, dir: str, colors: torch.Tensor = None):
    # Convert all tensors to numpy arrays in one go
    print(f"Saving ply to {dir}")
    numpy_data = {k: v.detach().cpu().numpy() for k, v in splats.items()}

    means = numpy_data["means"]
    scales = numpy_data["scales"]
    quats = numpy_data["quats"]
    opacities = numpy_data["opacities"]
    ply_data = {
        "positions": o3d.core.Tensor(means, dtype=o3d.core.Dtype.Float32),
        "normals": o3d.core.Tensor(np.zeros_like(means), dtype=o3d.core.Dtype.Float32),
        "opacity": o3d.core.Tensor(
            opacities.reshape(-1, 1), dtype=o3d.core.Dtype.Float32
        ),
    }

    if colors is not None:
        color = colors.detach().cpu().numpy().copy()  #
        for j in range(color.shape[1]):
            # Needs to be converted to shs as that's what all viewers take.
            ply_data[f"f_dc_{j}"] = o3d.core.Tensor(
                (color[:, j : j + 1] - 0.5) / 0.2820947917738781,
                dtype=o3d.core.Dtype.Float32,
                )
    else:
        sh0 = numpy_data["sh0"].transpose(0, 2, 1).reshape(means.shape[0], -1).copy()
        shN = numpy_data["shN"].transpose(0, 2, 1).reshape(means.shape[0], -1).copy()

        # Add sh0 and shN data
        for i, data in enumerate([sh0, shN]):
            prefix = "f_dc" if i == 0 else "f_rest"
            for j in range(data.shape[1]):
                ply_data[f"{prefix}_{j}"] = o3d.core.Tensor(
                    data[:, j : j + 1], dtype=o3d.core.Dtype.Float32
                )

    # Add scales and quats data
    for name, data in [("scale", scales), ("rot", quats)]:
        for i in range(data.shape[1]):
            ply_data[f"{name}_{i}"] = o3d.core.Tensor(
                data[:, i : i + 1], dtype=o3d.core.Dtype.Float32
            )

    pcd = o3d.t.geometry.PointCloud(ply_data)

    success = o3d.t.io.write_point_cloud(dir, pcd)
    assert success, "Could not save ply file."

# Command line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Merge multiple PLY files into a single PLY file.')
    parser.add_argument('input_dir', type=str, help='Directory containing input PLY files')
    parser.add_argument('output_path', type=str, help='Path where to save the merged PLY file')
    parser.add_argument('--pattern', type=str, default='*.ply',
                        help='Glob pattern to match PLY files (default: *.ply)')

    args = parser.parse_args()

    process_plys(args.input_dir, args.output_path, args.pattern)