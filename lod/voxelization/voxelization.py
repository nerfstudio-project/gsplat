import torch
import os
import numpy as np
from .voxel_methods import Voxel, AggregationMethod
# from .voxel_methods_others import Voxel, AggregationMethod
import time
import json

def read_gs_frame(gs_dir, iter):
    ckpt_path = os.path.join(gs_dir, f"ckpts/ckpt_{iter - 1}_rank0.pt")
    print(f"Reading gsplat frame, ckpt: {ckpt_path} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    return ckpt

class Voxelization:
    def __init__(self, voxel_size, bounds=(-4.0, 4.0, -4.0, 4.0, -4.0, 4.0)):
        """
        Initialize Voxelization class
        Args:
            voxel_size: Size of each voxel
            bounds: Tuple of (x_min, x_max, y_min, y_max, z_min, z_max) in meters
        """
        self.voxel_size : float = voxel_size
        self.bounds : tuple = bounds
        self.trained_gs_model : dict = None
        self.voxelized_gs_model : dict = None
        self.voxel_list : list[Voxel] = []
        self.voxel_occupancy_list : list[bool] = []
        self.voxel_grid : dict[tuple[int, int, int], Voxel] = {}  # Dictionary to store voxel indices for quick lookup
        self.device : str = "cuda" if torch.cuda.is_available() else "cpu"
        self.time : dict = {}
        
        # Calculate number of voxels in each dimension
        self.n_voxels_x : int = int((bounds[1] - bounds[0]) / voxel_size)
        self.n_voxels_y : int = int((bounds[3] - bounds[2]) / voxel_size)
        self.n_voxels_z : int = int((bounds[5] - bounds[4]) / voxel_size)
        
        print(f"Voxel grid dimensions: {self.n_voxels_x} x {self.n_voxels_y} x {self.n_voxels_z}")

    def read_gs(self, trained_model):
        """Read the trained gaussian splat model"""
        self.trained_gs_model = trained_model
        return self.trained_gs_model

    def _is_point_in_bounds(self, point):
        """Check if a point is within the specified bounds"""
        x, y, z = point
        return (self.bounds[0] <= x <= self.bounds[1] and
                self.bounds[2] <= y <= self.bounds[3] and
                self.bounds[4] <= z <= self.bounds[5])

    def _get_voxel_index(self, point):
        """Convert 3D point to voxel index"""
        if not self._is_point_in_bounds(point):
            return None
            
        # Convert to voxel coordinates
        voxel_x = int((point[0] - self.bounds[0]) / self.voxel_size)
        voxel_y = int((point[1] - self.bounds[2]) / self.voxel_size)
        voxel_z = int((point[2] - self.bounds[4]) / self.voxel_size)
        
        # Ensure indices are within bounds
        voxel_x = max(0, min(voxel_x, self.n_voxels_x - 1))
        voxel_y = max(0, min(voxel_y, self.n_voxels_y - 1))
        voxel_z = max(0, min(voxel_z, self.n_voxels_z - 1))
        
        return (voxel_x, voxel_y, voxel_z)

    def _create_voxel_grid(self):
        """Create voxel grid from gaussian means"""
        means = self.trained_gs_model["splats"]["means"].cpu().numpy()
        
        # Clear previous voxel data
        self.voxel_grid.clear()
        self.voxel_list.clear()
        self.voxel_occupancy_list.clear()
        
        # Create voxels for each gaussian
        points_outside_bounds = 0
        for i, mean in enumerate(means):
            start_time = time.time()
            # Get voxel index for the gaussian mean
            voxel_idx = self._get_voxel_index(mean)
            
            if voxel_idx is None:
                points_outside_bounds += 1
                continue
                
            if voxel_idx not in self.voxel_grid:
                # Calculate voxel center
                voxel_center = [
                    self.bounds[0] + (voxel_idx[0] + 0.5) * self.voxel_size,
                    self.bounds[2] + (voxel_idx[1] + 0.5) * self.voxel_size,
                    self.bounds[4] + (voxel_idx[2] + 0.5) * self.voxel_size
                ]
                
                voxel = Voxel(len(self.voxel_list), self.voxel_size, voxel_center)
                self.voxel_list.append(voxel)
                self.voxel_grid[voxel_idx] = voxel
                self.voxel_occupancy_list.append(True)
            
            self.voxel_grid[voxel_idx].add_gaussian(i)
            
        if points_outside_bounds > 0:
            print(f"Warning: {points_outside_bounds} points were outside the specified bounds")

    def voxelize(self, aggregate_method):
        """Perform voxelization of the gaussian splat model"""
        if self.trained_gs_model is None:
            raise ValueError("No trained model loaded. Call read_gs() first.")

        # Move model to CPU and convert to numpy arrays
        cpu_model = {
            "splats": {
                key: value.cpu().numpy() if torch.is_tensor(value) else value
                for key, value in self.trained_gs_model["splats"].items()
            }
        }

        # Create voxel grid
        print("Creating voxel grid...")
        start_time = time.time()
        self._create_voxel_grid()
        self.time["voxel_creation"] = (time.time() - start_time) * 1000.0
        print(f"Voxel grid created in time {self.time['voxel_creation']:.2f} ms")
        
        # Aggregate attributes for each voxel
        print("Aggregating attributes for each voxel...")
        start_time = time.time()
        for voxel in self.voxel_list:
            voxel.aggregate_attributes(cpu_model, aggregate_method=aggregate_method)
        self.time["attribute_aggregation"] = (time.time() - start_time) * 1000.0
        print(f"Attributes aggregated in time {self.time['attribute_aggregation']:.2f} ms")
        
        # Create voxelized model and convert back to PyTorch tensors
        print("Creating voxelized model...")
        start_time = time.time()
        self.voxelized_gs_model = {
            "splats": {
                "means": torch.from_numpy(np.stack([v.aggregated_mean for v in self.voxel_list])),
                "opacities": torch.from_numpy(np.stack([v.aggregated_opacity for v in self.voxel_list])),
                "quats": torch.from_numpy(np.stack([v.aggregated_quat for v in self.voxel_list])),
                "scales": torch.from_numpy(np.stack([v.aggregated_scale for v in self.voxel_list])),
                "sh0": torch.from_numpy(np.stack([v.aggregated_sh0 for v in self.voxel_list])),
                "shN": torch.from_numpy(np.stack([v.aggregated_shN for v in self.voxel_list]))
            },
            "step": 0           # After voxelization, step is reset to 0
        }
        # # Copy all other keys from trained model to voxelized model
        # for key, value in self.trained_gs_model.items():
        #     if key not in self.voxelized_gs_model:
        #         self.voxelized_gs_model[key] = value
        
        self.time["model_creation"] = (time.time() - start_time) * 1000.0
        print(f"Voxelized model created in time {self.time['model_creation']:.2f} ms")

        return self.voxelized_gs_model

    def save_voxelized_gs(self, save_path):
        """Save the voxelized gaussian splat model"""
        if self.voxelized_gs_model is None:
            raise ValueError("No voxelized model available. Call voxelize() first.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.voxelized_gs_model, save_path)
        print(f"Saved voxelized model to {save_path}")
        
    def voxel_stats(self, save_dir="./"):
        """Save time statistics to a json file

        Args:
            save_dir (str, optional): Path to save the json file. Defaults to "./".
        """
        with open(os.path.join(save_dir, "voxel_stats.json"), "w") as f:
            json.dump(self.time, f)

if __name__ == "__main__":
    # Example usage
    seq_name = "171026_pose3_no_ground"
    trained_voxel_size = 0.001
    frame_id = 580
    trained_iter = 10000
    trained_model_path = f"/ssd1/rajrup/Project/livogs_mv/results/nogop_voxel_size_{trained_voxel_size}_freeze_means/{seq_name}/{frame_id}/"
    
    # Read trained model
    trained_model = read_gs_frame(trained_model_path, trained_iter)
    
    # Test some effective methods
    # aggregate_methods = [
    #     AggregationMethod.scales_voxel_width
    # ]
    aggregate_methods = [
                            AggregationMethod.sum,  # Sum of gaussians
                            AggregationMethod.dominant,  # Dominant gaussian
                            AggregationMethod.mean,  # Mean of gaussians
                            AggregationMethod.kl_moment,  # KL(p||q) first‑/second‑moment match
                            AggregationMethod.kl_moment_hybrid_mean_moment,  # KL(p||q) for 1st moment
                            AggregationMethod.h3dgs,  # H3DGS aggregation method
                            AggregationMethod.h3dgs_hybrid_mean_moment,  # H3DGS for 1st moment
                            AggregationMethod.w2  # Wasserstein-2 distance aggregation method
                        ]
    
    # Test all other methods
    # aggregate_methods = [
    #     AggregationMethod.geometric,
    #     AggregationMethod.adaptive,
    #     AggregationMethod.covariance_weighted,  # Mathematical covariance combination
    #     AggregationMethod.gaussian_mixture,  # Proper Gaussian mixture reduction
    #     AggregationMethod.opacity_preserved,  # Focus on opacity preservation
    #     AggregationMethod.hybrid_dominant,  # Hybrid approach
    #     AggregationMethod.volume_weighted  # Volume-based weighting
    # ]
    print(f"Aggregate methods: {aggregate_methods}")
    
    # Voxelize at different resolutions
    voxel_sizes = [0.01, 0.025, 0.05, 0.1]
    for voxel_size in voxel_sizes:
        print(f"\nVoxelizing at size: {voxel_size}")
        # Initialize with 8m x 8m x 8m bounds centered at origin
        voxelizer = Voxelization(voxel_size, bounds=(-4.0, 4.0, -4.0, 4.0, -4.0, 4.0))
        voxelizer.read_gs(trained_model)
        for aggregate_method in aggregate_methods:
            print(f"\nVoxelizing with aggregate method: {aggregate_method}")
            voxelized_model = voxelizer.voxelize(aggregate_method)
        
            # Save voxelized model
            save_path = f"/ssd1/rajrup/Project/livogs_mv/results/nogop_voxel_size_{trained_voxel_size}_freeze_means_voxelized_to_voxel_size_{voxel_size}/{seq_name}/{frame_id}/{aggregate_method}/"
            save_model_path = os.path.join(save_path, "ckpts/ckpt_0_rank0.pt")
            voxelizer.save_voxelized_gs(save_model_path)
            
            # Save voxelization statistics
            voxelizer.voxel_stats(save_dir=save_path)
            
            # Print statistics
            print(f"Original gaussians: {len(trained_model['splats']['means'])}")
            print(f"Voxelized gaussians: {len(voxelized_model['splats']['means'])}")

'''
python examples/voxelization.py
'''