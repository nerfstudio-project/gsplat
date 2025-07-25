# Camera Bounds for Voxelization

This directory contains scripts to calculate tight bounds of 3D space captured by cameras for voxelization of the ActorsHQ dataset.

## Overview

The camera bounds functionality helps you determine the optimal voxel grid bounds for voxelization by analyzing the camera positions and viewing frustums from COLMAP data. This is particularly useful for the ActorsHQ dataset which contains 160 static cameras arranged around a performance stage.

## Files

- `camera_bounds.py` - Main script to calculate camera bounds from COLMAP data
- `voxelize_actorshq.py` - Example script showing how to use camera bounds for voxelization
- `bounds/` - Directory containing pre-calculated bounds (created after running `camera_bounds.py`)

## Quick Start

### 1. Calculate Camera Bounds

First, run the camera bounds script to analyze the ActorsHQ dataset:

```bash
python lod/camera_bounds.py
```

This will:
- Load COLMAP data from `data/Actor01/Sequence1/0/resolution_4`
- Calculate bounds using three different methods
- Save the results to JSON files in `lod/bounds/`

### 2. Use Bounds for Voxelization

```python
from lod.camera_bounds import get_actorshq_bounds
from lod.voxelization import Voxelization
from lod.voxel import AggregationMethod

# Get pre-calculated bounds
bounds = get_actorshq_bounds("viewing_frustum_bounds")

# Create voxelizer with camera bounds
voxelizer = Voxelization(voxel_size=0.05, bounds=bounds)

# Load and voxelize your model
voxelizer.read_gs(trained_model)
voxelized_model = voxelizer.voxelize(AggregationMethod.h3dgs)
```

### 3. Run the Example

```bash
python lod/voxelize_actorshq.py
```

## Bounds Methods

The script provides three methods for calculating bounds:

### 1. Camera Position Bounds (`camera_position_bounds`)
- **What it does**: Calculates tight bounds around camera positions only
- **Use case**: Minimal coverage, good for debugging
- **Size**: ~5.6 x 3.7 x 5.8 meters
- **Volume**: ~119 cubic meters

### 2. Viewing Frustum Bounds (`viewing_frustum_bounds`) - **Recommended**
- **What it does**: Includes the actual viewing volume of all cameras
- **Use case**: Covers the entire 3D stage that cameras can see
- **Size**: ~7.7 x 7.0 x 7.7 meters
- **Volume**: ~417 cubic meters

### 3. Scene Center Bounds (`scene_center_bounds`)
- **What it does**: Creates symmetric bounds around the scene center
- **Use case**: Uniform coverage, good for certain algorithms
- **Size**: ~9.0 x 9.0 x 9.0 meters
- **Volume**: ~727 cubic meters

## Method Comparison

Here's a detailed comparison to help you choose the right bounds method:

| Aspect | Camera Position | Viewing Frustum | Scene Center |
|--------|-----------------|-----------------|--------------|
| **Coverage** | Camera locations only | Actual viewing volume | Symmetric around center |
| **Size** | Smallest (~20 m³) | Medium (~774 m³) | Largest (~727 m³) |
| **Shape** | Irregular (camera arrangement) | Irregular (viewing volume) | Cubic/symmetric |
| **Computation** | Fastest | Medium | Fast |
| **Memory Usage** | Lowest | Medium | Highest |
| **Accuracy** | Cameras only | Scene content | Over-inclusive |

### When to Use Each Method

**🎯 Camera Position Bounds** - Use when:
- Debugging camera placement
- Analyzing camera rig geometry
- Minimal memory requirements
- **⚠️ NOT recommended for voxelization**

**🎯 Viewing Frustum Bounds** - Use when:
- Voxelizing scene content (recommended)
- Want to cover exactly what cameras see
- Balancing accuracy vs. memory usage
- Working with performance/actor data

**🎯 Scene Center Bounds** - Use when:
- Need symmetric/uniform coverage
- Algorithm requires cubic bounds
- Want to ensure no content is missed
- Memory is not a constraint

### Visual Representation

```
Camera Position:     Viewing Frustum:     Scene Center:
     📷                   📷 ╲                 ┌─────────┐
   📷   📷       vs     📷   📷 ╲     vs      │  📷 ╲   │
     📷                   📷 ╱    ╲           │📷   📷 ╲│
                                  ╲           │  📷 ╱   │
[Tight around        [Covers actual       [Symmetric
 cameras]             viewing volume]       coverage]
```

### Performance Impact

For ActorsHQ dataset voxelization with 0.05m voxels:

| Method | Voxel Grid Size | Total Voxels | Memory Est. |
|--------|-----------------|--------------|-------------|
| Camera Position | ~44×60×60 | ~158K | ~25 MB |
| Viewing Frustum | ~176×188×188 | ~6.2M | ~1.0 GB |
| Scene Center | ~180×180×180 | ~5.8M | ~0.9 GB |

**💡 Recommendation**: Use **Viewing Frustum Bounds** for voxelization as it provides the best balance of accuracy and efficiency.

## ActorsHQ Dataset Results

For the ActorsHQ dataset (160 cameras):
- **Scene Center**: (0.004, 1.251, 0.083)
- **Recommended Voxel Sizes**:
  - Coarse (32³): 0.24 meters
  - Medium (64³): 0.12 meters
  - Fine (128³): 0.06 meters
  - Ultra (256³): 0.03 meters

## API Reference

### Functions

#### `get_actorshq_bounds(method="viewing_frustum_bounds")`
Load pre-calculated bounds for ActorsHQ dataset.

**Parameters:**
- `method` (str): Bounds calculation method
  - `"camera_position_bounds"` - Minimal coverage
  - `"viewing_frustum_bounds"` - Viewing volume (recommended)
  - `"scene_center_bounds"` - Symmetric coverage

**Returns:**
- `Tuple[float, float, float, float, float, float]`: (x_min, x_max, y_min, y_max, z_min, z_max)

#### `load_bounds_from_json(json_path)`
Load bounds from a JSON file.

**Parameters:**
- `json_path` (str): Path to bounds JSON file

**Returns:**
- `Tuple[float, float, float, float, float, float]`: Bounds tuple

### Classes

#### `CameraBounds(data_dir, factor=1, normalize=False)`
Calculate bounds from COLMAP data.

**Methods:**
- `get_camera_bounds(padding=0.0)` - Simple camera position bounds
- `get_viewing_frustum_bounds(max_depth=5.0, padding=0.0)` - Viewing volume bounds
- `get_scene_center_bounds(radius_multiplier=1.5, padding=0.0)` - Scene center bounds
- `print_bounds_info(bounds)` - Print bounds information
- `save_bounds(bounds, output_path, method)` - Save bounds to JSON

## Integration with Existing Code

To use camera bounds in your existing voxelization code:

```python
# Before (hardcoded bounds)
voxelizer = Voxelization(voxel_size, bounds=(-4.0, 4.0, -4.0, 4.0, -4.0, 4.0))

# After (camera-based bounds)
from lod.camera_bounds import get_actorshq_bounds
bounds = get_actorshq_bounds("viewing_frustum_bounds")
voxelizer = Voxelization(voxel_size, bounds=bounds)
```

## Troubleshooting

### "Bounds file not found" Error
Run `python lod/camera_bounds.py` first to calculate and save bounds.

### "COLMAP directory does not exist" Error
Make sure the ActorsHQ dataset is placed in `data/Actor01/Sequence1/0/resolution_4/`.

### Memory Issues with Fine Voxels
Use coarser voxel sizes (0.1-0.2 meters) for initial testing, then gradually reduce.

## Dataset Requirements

- **Directory structure**: `data/Actor01/Sequence1/0/resolution_4/`
- **Required files**: 
  - `sparse/0/cameras.bin` (or `.txt`)
  - `sparse/0/images.bin` (or `.txt`)
  - `sparse/0/points3D.bin` (or `.txt`)
  - `images/` directory

## Performance Notes

- **Viewing frustum calculation**: ~1-2 seconds for 160 cameras
- **Voxel grid creation**: Depends on voxel size and scene size
- **Memory usage**: Scales with number of voxels (O(n³))

## Example Output

```
Loading COLMAP data from: data/Actor01/Sequence1/0/resolution_4
Found 160 cameras

=== Method 2: Viewing Frustum Bounds ===
Bounds Information:
X: [-3.865, 3.822] (size: 7.687)
Y: [-2.602, 4.438] (size: 7.040)
Z: [-3.778, 3.926] (size: 7.704)
Center: (-0.021, 0.918, 0.074)
Volume: 416.962 cubic meters
```

This provides optimal bounds for voxelization that cover the entire 3D stage captured by the 160 static cameras in the ActorsHQ dataset. 