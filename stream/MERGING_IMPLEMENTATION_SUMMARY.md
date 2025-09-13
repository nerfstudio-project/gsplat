# Gaussian Merging Implementation Summary

## Overview

This document summarizes the implementation of Gaussian merging for 3D Gaussian Splatting, which extends beyond simple culling to intelligently combine small Gaussians while preserving visual quality. The implementation is designed to be modular and extensible for research purposes.

## Background

Instead of simply removing small Gaussians (culling), merging combines nearby small Gaussians into single, appropriately-sized Gaussians. This approach:
- **Preserves information** that would be lost with culling
- **Maintains spatial coverage** of the scene
- **Provides adaptive level-of-detail** based on viewing distance
- **Potentially improves quality** by creating properly-sized Gaussians

## Parameter Space Requirements

**IMPORTANT**: All merging functions expect Gaussian parameters in **linear space**:
- **Scales**: Linear values (not log space)
- **Opacities**: Linear values (not logit space)

This is automatically handled when loading checkpoints using `load_checkpoint()` from evaluation scripts, which converts parameters from parameter space to linear space before passing to merging functions.

## Implementation Architecture

### Core Components

#### 1. **Merge Candidate Identification** (`find_merge_candidates`)
- Identifies small Gaussians based on pixel size or pixel area thresholds
- Reuses existing pixel size/area calculation from culling module
- Supports both pixel size (`< 2.0 pixels`) and pixel area (`< π pixels²`) criteria

#### 2. **Clustering System** (`cluster_gaussians`)
Extensible clustering framework supporting multiple algorithms:

**K-Nearest Neighbors (KNN)**
- Builds adjacency graph based on spatial proximity
- Uses connected components to form clusters
- Parameters: `k_neighbors`, `max_distance`, `min_cluster_size`

**DBSCAN**
- Density-based clustering for arbitrary cluster shapes
- Handles noise and varying densities
- Parameters: `eps`, `min_samples`

**Distance-Based**
- Simple distance threshold clustering
- Fast and straightforward
- Parameters: `max_distance`, `min_cluster_size`

#### 3. **Merging Strategies** (`merge_cluster`)
Multiple mathematical approaches for combining Gaussians:

**Weighted Mean**
- Opacity-weighted parameter averaging
- Simple and fast
- Parameters: `weight_by_opacity`

**Moment Matching**
- Preserves statistical properties of Gaussian mixture
- Can preserve total volume
- Parameters: `preserve_volume`

#### 4. **Main Pipeline** (`merge_gaussians`)
- Orchestrates the complete merging process
- Supports iterative merging with convergence criteria
- Provides comprehensive statistics and logging

## Mathematical Foundation

### Weighted Mean Merging
For N Gaussians with parameters {μᵢ, qᵢ, sᵢ, αᵢ, cᵢ} and weights wᵢ = αᵢ:

```
μ_merged = Σᵢ wᵢ μᵢ / Σᵢ wᵢ
q_merged = normalize(Σᵢ wᵢ qᵢ)
s_merged = Σᵢ wᵢ sᵢ / Σᵢ wᵢ
α_merged = clamp(Σᵢ αᵢ, 0, 1)
c_merged = Σᵢ wᵢ cᵢ / Σᵢ wᵢ
```

### Moment Matching (Advanced)
Preserves first and second moments of the Gaussian mixture:
- First moment (mean): Same as weighted mean
- Second moment: Accounts for covariance between Gaussians
- Volume preservation: Redistributes total volume spherically

## API Usage

### Basic Usage
```python
from stream.merging import merge_gaussians

# Merge small Gaussians using KNN clustering and weighted mean
merged_means, merged_quats, merged_scales, merged_opacities, merged_colors, info = merge_gaussians(
    means, quats, scales, opacities, colors,
    viewmat, K, width, height,
    pixel_size_threshold=2.0,
    clustering_method="knn",
    merge_strategy="weighted_mean"
)

print(f"Merged {info['original_count']} → {info['final_count']} Gaussians")
print(f"Reduction: {info['reduction_ratio']:.1%}")
```

### Advanced Configuration
```python
# Custom clustering and merging parameters
merged_params, info = merge_gaussians(
    # ... Gaussian parameters ...
    clustering_method="dbscan",
    merge_strategy="moment_matching",
    clustering_kwargs={
        "eps": 0.15,
        "min_samples": 3
    },
    merge_kwargs={
        "preserve_volume": True
    },
    max_iterations=3,
    min_reduction_ratio=0.01
)
```

### Using Different Metrics
```python
# Use pixel area instead of pixel size
merged_params, info = merge_gaussians(
    # ... parameters ...
    use_pixel_area=True,
    pixel_area_threshold=4.0  # π pixels²
)
```

## Evaluation Framework

### Evaluation Script
`scripts/merging/evaluate_static_merging.py` provides comprehensive quality assessment:

**Merging Configurations Tested:**
- KNN + Weighted Mean
- DBSCAN + Weighted Mean  
- KNN + Moment Matching
- Frustum Culling + KNN Merging

**Quality Metrics:**
- PSNR, SSIM, LPIPS (standard image quality)
- Masked metrics (focus on valid regions)
- Cropped LPIPS (perceptual quality in content regions)
- Merging statistics (reduction ratios, iteration counts)

### Batch Evaluation
```bash
# Run evaluation across multiple thresholds
./scripts/merging/evaluate_quality_of_merging.sh
```

## Performance Characteristics

### Computational Complexity
- **Candidate Finding**: O(N) - linear in number of Gaussians
- **KNN Clustering**: O(N log N) - dominated by nearest neighbor search
- **DBSCAN Clustering**: O(N log N) - average case with spatial indexing
- **Merging**: O(C) - linear in number of clusters

### Memory Usage
- **Temporary Storage**: Clustering requires copying candidate positions
- **Final Output**: Reduced memory usage due to fewer Gaussians
- **Iterative Merging**: Constant memory overhead per iteration

### Scalability
- **Small Scenes** (< 10K Gaussians): All methods perform well
- **Medium Scenes** (10K-100K): KNN and distance-based preferred
- **Large Scenes** (> 100K): Distance-based clustering recommended

## Validation Results

### Test Suite (`stream/tests/test_merging.py`)
✅ **All tests passing:**
- Merge candidate identification
- All clustering methods (KNN, DBSCAN, distance-based)
- Both merging strategies (weighted mean, moment matching)
- Full pipeline integration
- Edge case handling (small datasets, no candidates)

### Key Findings
- **Clustering Success**: All methods successfully identify spatial clusters
- **Parameter Consistency**: Merged Gaussians maintain valid properties
- **Quaternion Handling**: Proper normalization and rotation averaging
- **Edge Case Robustness**: Graceful handling of degenerate cases

## Research Extensions

### Easy Extensions
1. **New Clustering Methods**: Add to `cluster_gaussians` switch statement
2. **New Merging Strategies**: Add to `merge_cluster` switch statement  
3. **Different Distance Metrics**: Modify clustering distance calculations
4. **Adaptive Thresholds**: Distance-dependent pixel size thresholds

### Advanced Research Directions
1. **Hierarchical Merging**: Multi-scale clustering and merging
2. **Content-Aware Merging**: Weight by color/texture similarity
3. **Temporal Merging**: Merging across video frames
4. **Quality-Guided Merging**: Use rendering quality as merging criterion

## Integration with Existing Code

### Culling Integration
- Reuses pixel size/area calculation from `stream.culling`
- Can be combined with frustum culling
- Drop-in replacement for distance culling

### Rendering Pipeline
- Compatible with existing rasterization
- Produces standard Gaussian parameters
- Maintains SH color support

## Configuration Recommendations

### For Quality Priority
```python
clustering_method="knn"
merge_strategy="moment_matching"
clustering_kwargs={"k_neighbors": 5, "max_distance": 0.05, "min_cluster_size": 3}
merge_kwargs={"preserve_volume": True}
max_iterations=3
```

### For Performance Priority
```python
clustering_method="distance_based"
merge_strategy="weighted_mean"
clustering_kwargs={"max_distance": 0.1, "min_cluster_size": 2}
max_iterations=1
```

### For Research/Exploration
```python
clustering_method="dbscan"
merge_strategy="moment_matching"
clustering_kwargs={"eps": 0.1, "min_samples": 2}
merge_kwargs={"preserve_volume": False}
max_iterations=5
min_reduction_ratio=0.005
```

## Conclusion

The Gaussian merging implementation provides a sophisticated alternative to simple culling, with:

- **Modular Design**: Easy to extend with new clustering and merging methods
- **Research Ready**: Comprehensive evaluation framework included
- **Production Viable**: Efficient implementation with proper error handling
- **Quality Focused**: Multiple strategies to preserve visual fidelity

This foundation enables extensive research into advanced Gaussian Splatting optimization techniques while maintaining compatibility with existing rendering pipelines.
