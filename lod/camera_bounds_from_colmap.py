import os
import sys
import numpy as np
import json
from typing import Tuple

# Add the parent directory to the path to import from examples
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from examples.datasets.colmap import Parser


def load_bounds_from_json(json_path: str) -> Tuple[float, float, float, float, float, float]:
    """
    Load bounds from a JSON file saved by CameraBounds.
    
    Args:
        json_path: Path to the JSON file containing bounds
        
    Returns:
        Tuple of (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return tuple(data['bounds_tuple'])


def get_actorshq_bounds(method: str = "viewing_frustum_bounds") -> Tuple[float, float, float, float, float, float]:
    """
    Get the pre-calculated bounds for ActorsHQ dataset.
    
    Args:
        method: Method to use for bounds calculation
               Options: "camera_position_bounds", "viewing_frustum_bounds", "scene_center_bounds"
               
    Returns:
        Tuple of (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    bounds_dir = os.path.join(os.path.dirname(__file__), "bounds")
    json_path = os.path.join(bounds_dir, f"{method}.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Bounds file not found: {json_path}. Please run this script first to calculate bounds.")
    
    return load_bounds_from_json(json_path)


class CameraBounds:
    """Calculate tight bounds of 3D space captured by cameras from COLMAP data."""
    
    def __init__(self, data_dir: str, factor: int = 1, normalize: bool = False):
        """
        Initialize CameraBounds with COLMAP data.
        
        Args:
            data_dir: Path to the COLMAP data directory
            factor: Downsampling factor for images
            normalize: Whether to normalize the camera poses
        """
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        
        # Parse COLMAP data
        print(f"Loading COLMAP data from: {data_dir}")
        self.parser = Parser(
            data_dir=data_dir,
            factor=factor,
            normalize=normalize,
            test_every=1  # Don't skip any cameras for bounds calculation
        )
        
        # Extract camera positions
        self.camera_positions = self.parser.camtoworlds[:, :3, 3]  # (N, 3)
        self.camera_orientations = self.parser.camtoworlds[:, :3, :3]  # (N, 3, 3)
        
        print(f"Found {len(self.camera_positions)} cameras")
        
    def get_camera_bounds(self, padding: float = 0.0) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate tight bounds from camera positions.
        
        Args:
            padding: Additional padding to add around the bounds (in meters)
            
        Returns:
            Tuple of (x_min, x_max, y_min, y_max, z_min, z_max)
        """
        # Calculate bounds from camera positions
        min_pos = np.min(self.camera_positions, axis=0)
        max_pos = np.max(self.camera_positions, axis=0)
        
        # Add padding
        x_min, y_min, z_min = min_pos - padding
        x_max, y_max, z_max = max_pos + padding
        
        return (x_min, x_max, y_min, y_max, z_min, z_max)
    
    def get_viewing_frustum_bounds(self, max_depth: float = 5.0, padding: float = 0.0) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate bounds that include the viewing frustum of all cameras.
        
        Args:
            max_depth: Maximum depth to consider for the viewing frustum (in meters)
            padding: Additional padding to add around the bounds (in meters)
            
        Returns:
            Tuple of (x_min, x_max, y_min, y_max, z_min, z_max)
        """
        frustum_points = []
        
        # For each camera, calculate frustum corners
        for i, (pos, orientation) in enumerate(zip(self.camera_positions, self.camera_orientations)):
            camera_id = self.parser.camera_ids[i]
            
            if camera_id not in self.parser.Ks_dict:
                continue
                
            K = self.parser.Ks_dict[camera_id]
            width, height = self.parser.imsize_dict[camera_id]
            
            # Get camera intrinsics
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # Define image plane corners
            corners_2d = np.array([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height]
            ])
            
            # Project corners to 3D at max_depth
            for corner in corners_2d:
                # Convert to normalized camera coordinates
                x_norm = (corner[0] - cx) / fx
                y_norm = (corner[1] - cy) / fy
                
                # Create ray direction in camera space
                ray_dir_cam = np.array([x_norm, y_norm, 1.0])
                ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)
                
                # Transform to world space
                ray_dir_world = orientation @ ray_dir_cam
                
                # Calculate point at max_depth
                point_world = pos + ray_dir_world * max_depth
                frustum_points.append(point_world)
        
        # Include camera positions in the frustum points
        all_points = np.vstack([self.camera_positions, np.array(frustum_points)])
        
        # Calculate bounds
        min_pos = np.min(all_points, axis=0)
        max_pos = np.max(all_points, axis=0)
        
        # Add padding
        x_min, y_min, z_min = min_pos - padding
        x_max, y_max, z_max = max_pos + padding
        
        return (x_min, x_max, y_min, y_max, z_min, z_max)
    
    def get_scene_center_bounds(self, radius_multiplier: float = 1.5, padding: float = 0.0) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate bounds based on scene center and camera distances.
        
        Args:
            radius_multiplier: Multiplier for the maximum camera distance from center
            padding: Additional padding to add around the bounds (in meters)
            
        Returns:
            Tuple of (x_min, x_max, y_min, y_max, z_min, z_max)
        """
        # Calculate scene center
        scene_center = np.mean(self.camera_positions, axis=0)
        
        # Calculate maximum distance from center
        distances = np.linalg.norm(self.camera_positions - scene_center, axis=1)
        max_distance = np.max(distances) * radius_multiplier
        
        # Create cubic bounds around the center
        x_min = scene_center[0] - max_distance - padding
        x_max = scene_center[0] + max_distance + padding
        y_min = scene_center[1] - max_distance - padding
        y_max = scene_center[1] + max_distance + padding
        z_min = scene_center[2] - max_distance - padding
        z_max = scene_center[2] + max_distance + padding
        
        return (x_min, x_max, y_min, y_max, z_min, z_max)
    
    def print_bounds_info(self, bounds: Tuple[float, float, float, float, float, float]):
        """Print information about the calculated bounds."""
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        print(f"\nBounds Information:")
        print(f"X: [{x_min:.3f}, {x_max:.3f}] (size: {x_max - x_min:.3f})")
        print(f"Y: [{y_min:.3f}, {y_max:.3f}] (size: {y_max - y_min:.3f})")
        print(f"Z: [{z_min:.3f}, {z_max:.3f}] (size: {z_max - z_min:.3f})")
        print(f"Center: ({(x_min + x_max)/2:.3f}, {(y_min + y_max)/2:.3f}, {(z_min + z_max)/2:.3f})")
        
        # Calculate volume
        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        print(f"Volume: {volume:.3f} cubic meters")
        
    def save_bounds(self, bounds: Tuple[float, float, float, float, float, float], 
                   output_path: str, method: str = "camera_bounds"):
        """
        Save bounds to a JSON file.
        
        Args:
            bounds: The bounds tuple
            output_path: Path to save the JSON file
            method: Method used to calculate bounds
        """
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        bounds_data = {
            "method": method,
            "bounds": {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
                "z_min": float(z_min),
                "z_max": float(z_max)
            },
            "bounds_tuple": list(bounds),
            "center": [
                float((x_min + x_max) / 2),
                float((y_min + y_max) / 2),
                float((z_min + z_max) / 2)
            ],
            "size": [
                float(x_max - x_min),
                float(y_max - y_min),
                float(z_max - z_min)
            ],
            "volume": float((x_max - x_min) * (y_max - y_min) * (z_max - z_min)),
            "num_cameras": int(len(self.camera_positions)),
            "data_dir": self.data_dir
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(bounds_data, f, indent=2)
        
        print(f"Bounds saved to: {output_path}")


def main():
    """Main function to calculate and save camera bounds."""
    # Path to the ActorsHQ dataset
    data_dir = "data/Actor01/Sequence1/0/resolution_4"
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        print("Please make sure the ActorsHQ dataset is placed in the correct location.")
        return
    
    # Initialize camera bounds calculator
    bounds_calculator = CameraBounds(data_dir, factor=1, normalize=True)
    
    # Method 1: Simple camera position bounds
    print("\n=== Method 1: Camera Position Bounds ===")
    camera_bounds = bounds_calculator.get_camera_bounds(padding=0.5)
    bounds_calculator.print_bounds_info(camera_bounds)
    bounds_calculator.save_bounds(
        camera_bounds, 
        "lod/bounds/camera_position_bounds.json", 
        "camera_position_bounds"
    )
    
    # Method 2: Viewing frustum bounds
    print("\n=== Method 2: Viewing Frustum Bounds ===")
    frustum_bounds = bounds_calculator.get_viewing_frustum_bounds(max_depth=5.0, padding=0.5)
    bounds_calculator.print_bounds_info(frustum_bounds)
    bounds_calculator.save_bounds(
        frustum_bounds, 
        "lod/bounds/viewing_frustum_bounds.json", 
        "viewing_frustum_bounds"
    )
    
    # Method 3: Scene center bounds
    print("\n=== Method 3: Scene Center Bounds ===")
    scene_bounds = bounds_calculator.get_scene_center_bounds(radius_multiplier=1.5, padding=0.5)
    bounds_calculator.print_bounds_info(scene_bounds)
    bounds_calculator.save_bounds(
        scene_bounds, 
        "lod/bounds/scene_center_bounds.json", 
        "scene_center_bounds"
    )
    
    # Recommend the most suitable bounds
    print("\n=== Recommendations ===")
    print("- Use 'camera_position_bounds' for minimal coverage of camera positions")
    print("- Use 'viewing_frustum_bounds' for coverage of the actual viewing volume")
    print("- Use 'scene_center_bounds' for symmetric coverage around the scene center")
    print("\nFor voxelization, 'viewing_frustum_bounds' is typically the best choice.")


if __name__ == "__main__":
    main() 