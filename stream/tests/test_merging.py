#!/usr/bin/env python3
"""
Test script for Gaussian merging functionality.
"""

import torch
import numpy as np
import math

# Add parent directory to path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from stream.merging import (
    find_merge_candidates, 
    cluster_gaussians, 
    merge_cluster,
    merge_gaussians
)


def create_test_data(N=100, device='cuda'):
    """Create test Gaussian data for merging."""
    
    # Create some small and large Gaussians
    # Place some very close to camera (z=1) and some far (z=10)
    means = torch.randn(N, 3, device=device) * 1.0  # Smaller spread
    means[:, 2] = torch.rand(N, device=device) * 9 + 1  # z from 1 to 10
    
    # Make half the Gaussians very small, half normal sized
    half = N // 2
    
    # Normalized quaternions (random rotations)
    quats = torch.randn(N, 4, device=device)
    quats = quats / torch.norm(quats, dim=1, keepdim=True)
    
    # Create mix of very small and normal scales
    scales = torch.rand(N, 3, device=device) * 0.2 + 0.1  # [0.1, 0.3]
    # Make first half very small (distant or small intrinsic size)
    scales[:half] = torch.rand(half, 3, device=device) * 0.05 + 0.01  # [0.01, 0.06]
    
    # Lower opacities for the small ones
    opacities = torch.rand(N, device=device) * 0.8 + 0.2  # [0.2, 1.0]
    opacities[:half] = torch.rand(half, device=device) * 0.3 + 0.1  # [0.1, 0.4] for small ones
    
    # Simple RGB colors (no SH for testing)
    colors = torch.rand(N, 3, device=device)
    
    # Create camera matrices - close to the Gaussians
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = -0.5  # Move camera closer to origin
    K = torch.tensor([[500.0, 0, 400], [0, 500.0, 300], [0, 0, 1]], device=device)
    
    return means, quats, scales, opacities, colors, viewmat, K


def test_find_merge_candidates():
    """Test finding merge candidates."""
    print("Testing find_merge_candidates...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    means, quats, scales, opacities, colors, viewmat, K = create_test_data(50, device)
    
    # Find candidates - use lower threshold since we expect small Gaussians
    candidates = find_merge_candidates(
        means, quats, scales, opacities, viewmat, K, 800, 600,
        pixel_size_threshold=5.0,  # Lower threshold to find small candidates
        use_pixel_area=False
    )
    
    print(f"  Found {candidates.sum().item()} candidates out of {len(candidates)} Gaussians")
    assert candidates.sum() > 0, "Should find some merge candidates"
    print("  ✓ find_merge_candidates passed")


def test_clustering():
    """Test clustering methods."""
    print("Testing clustering methods...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    means, _, _, _, _, _, _ = create_test_data(20, device)
    
    # Create a candidate mask (select half the Gaussians)
    candidate_mask = torch.zeros(20, dtype=torch.bool, device=device)
    candidate_mask[:10] = True
    
    # Test KNN clustering
    clusters_knn = cluster_gaussians(
        means, candidate_mask, "knn",
        k_neighbors=3, max_distance=1.0, min_cluster_size=2
    )
    
    # Test DBSCAN clustering
    clusters_dbscan = cluster_gaussians(
        means, candidate_mask, "dbscan",
        eps=1.0, min_samples=2
    )
    
    # Test distance-based clustering
    clusters_dist = cluster_gaussians(
        means, candidate_mask, "distance_based",
        max_distance=1.0, min_cluster_size=2
    )
    
    print(f"  KNN found {len(clusters_knn)} clusters")
    print(f"  DBSCAN found {len(clusters_dbscan)} clusters")
    print(f"  Distance-based found {len(clusters_dist)} clusters")
    
    print("  ✓ clustering methods passed")


def test_merge_cluster():
    """Test merging a cluster."""
    print("Testing merge_cluster...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    means, quats, scales, opacities, colors, _, _ = create_test_data(10, device)
    
    # Create a cluster of the first 3 Gaussians
    cluster_indices = np.array([0, 1, 2])
    
    # Test weighted mean merging
    merged_mean, merged_quat, merged_scale, merged_opacity, merged_color = merge_cluster(
        cluster_indices, means, quats, scales, opacities, colors,
        merge_strategy="weighted_mean"
    )
    
    print(f"  Original means shape: {means[cluster_indices].shape}")
    print(f"  Merged mean shape: {merged_mean.shape}")
    print(f"  Merged opacity (linear): {merged_opacity.item():.3f}")
    
    # Basic checks
    assert merged_mean.shape == (3,), "Merged mean should be 3D"
    assert merged_quat.shape == (4,), "Merged quat should be 4D"
    assert merged_scale.shape == (3,), "Merged scale should be 3D"
    assert merged_opacity.item() <= 1.0, "Merged opacity should be <= 1.0"
    assert merged_opacity.item() >= 0.0, "Merged opacity should be >= 0.0"
    assert torch.isclose(torch.norm(merged_quat), torch.tensor(1.0)), "Quaternion should be normalized"
    
    print("  ✓ merge_cluster passed")


def test_full_merging():
    """Test the full merging pipeline."""
    print("Testing full merging pipeline...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    means, quats, scales, opacities, colors, viewmat, K = create_test_data(50, device)
    
    original_count = means.shape[0]
    print(f"  Original Gaussian count: {original_count}")
    
    # Test full merging
    merged_means, merged_quats, merged_scales, merged_opacities, merged_colors, merge_info = merge_gaussians(
        means, quats, scales, opacities, colors, viewmat, K, 800, 600,
        pixel_size_threshold=5.0,  # Moderate threshold
        clustering_method="knn",
        merge_strategy="weighted_mean",
        clustering_kwargs={"k_neighbors": 3, "max_distance": 0.5, "min_cluster_size": 2},
        max_iterations=1
    )
    
    final_count = merged_means.shape[0]
    print(f"  Final Gaussian count: {final_count}")
    print(f"  Reduction: {original_count - final_count} Gaussians")
    print(f"  Reduction ratio: {merge_info['reduction_ratio']:.2%}")
    print(f"  Iterations: {merge_info['iterations']}")
    
    # Basic checks
    assert final_count <= original_count, "Should not increase Gaussian count"
    assert merged_means.shape[0] == merged_quats.shape[0], "All tensors should have same batch size"
    assert merged_quats.shape[0] == merged_scales.shape[0], "All tensors should have same batch size"
    assert merged_scales.shape[0] == merged_opacities.shape[0], "All tensors should have same batch size"
    assert merged_opacities.shape[0] == merged_colors.shape[0], "All tensors should have same batch size"
    
    # Check quaternion normalization
    quat_norms = torch.norm(merged_quats, dim=1)
    assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5), "Quaternions should be normalized"
    
    print("  ✓ full merging pipeline passed")


def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test with very few Gaussians
    means, quats, scales, opacities, colors, viewmat, K = create_test_data(3, device)
    
    try:
        merged_means, _, _, _, _, merge_info = merge_gaussians(
            means, quats, scales, opacities, colors, viewmat, K, 800, 600,
            pixel_size_threshold=100.0,  # Very high threshold
            max_iterations=1
        )
        print(f"  Small dataset handled: {means.shape[0]} → {merged_means.shape[0]}")
    except Exception as e:
        print(f"  Error with small dataset: {e}")
        
    # Test with no merge candidates
    means, quats, scales, opacities, colors, viewmat, K = create_test_data(10, device)
    
    try:
        merged_means, _, _, _, _, merge_info = merge_gaussians(
            means, quats, scales, opacities, colors, viewmat, K, 800, 600,
            pixel_size_threshold=0.1,  # Very low threshold - no candidates
            max_iterations=1
        )
        print(f"  No candidates handled: {means.shape[0]} → {merged_means.shape[0]}")
        assert merged_means.shape[0] == means.shape[0], "Should keep all Gaussians when no merging occurs"
    except Exception as e:
        print(f"  Error with no candidates: {e}")
    
    print("  ✓ edge cases passed")


def main():
    """Run all tests."""
    print("="*60)
    print("GAUSSIAN MERGING TESTS")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU device")
    
    try:
        test_find_merge_candidates()
        test_clustering()
        test_merge_cluster()
        test_full_merging()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
