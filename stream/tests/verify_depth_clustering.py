#!/usr/bin/env python3
"""
Verify CUDA depth clustering correctness (Stages 7-8)

This script verifies the correctness of:
1. Stage 7: Depth-based clustering within pixel groups
2. Stage 8: Result processing (cluster sizes, indices extraction, offsets)

It checks correctness properties rather than exact matching since depth clustering
can legitimately produce different valid solutions.
"""

import torch
from torch import Tensor
import numpy as np
from typing import Dict, List, Tuple, Set
import sys
import os
import json
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.config import Config, load_config_from_toml, merge_config
from scripts.utils import set_result_dir, load_checkpoint, load_poses
from stream.merging import find_merge_candidates
from stream.clustering_cuda import cluster_center_in_pixel_cuda, cluster_center_in_pixel_torch
from stream.cuda._wrapper import extract_pixel_groups_step2_cuda, extract_cluster_assignments_step7_cuda
from gsplat.cuda._wrapper import proj, world_to_cam

def load_test_data(pose_id: int = 80) -> Dict:
    """Load the same test data used in verification scripts"""
    cfg = Config()
    
    # Load template config
    template_path = "./configs/actorshq_stream.toml"
    config_from_file = load_config_from_toml(template_path)
    cfg = merge_config(cfg, config_from_file)
    
    # Set experiment parameters
    exp_name = f"actorshq_l1_{1.0 - cfg.ssim_lambda}_ssim_{cfg.ssim_lambda}"
    if cfg.masked_l1_loss:
        exp_name += f"_ml1_{cfg.masked_l1_lambda}"
    if cfg.masked_ssim_loss:
        exp_name += f"_mssim_{cfg.masked_ssim_lambda}"
    if cfg.alpha_loss:
        exp_name += f"_alpha_{cfg.alpha_lambda}"
    if cfg.scale_var_loss:
        exp_name += f"_svar_{cfg.scale_var_lambda}"
    if cfg.random_bkgd:
        exp_name += "_rbkgd"
    
    # Set data paths
    frame_id = 0
    cfg.data_dir = os.path.join(cfg.actorshq_data_dir, f"{frame_id}", f"resolution_{cfg.resolution}")
    cfg.exp_name = exp_name
    cfg.run_mode = "render"
    cfg.scene_id = frame_id
    
    set_result_dir(cfg, exp_name=exp_name)
    iter = cfg.max_steps
    ckpt = os.path.join(f"{cfg.result_dir}/ckpts/ckpt_{iter - 1}_rank0.pt")
    cfg.ckpt = ckpt
    
    log.info(f"Loading scene data for pose {pose_id}...")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    assert os.path.exists(cfg.ckpt), f"Checkpoint not found: {cfg.ckpt}"
    means, quats, scales, opacities, colors = load_checkpoint(cfg.ckpt, device)
    
    # Load poses
    poses_dir = os.path.join(cfg.result_dir, "viewer_poses")
    poses_file = os.path.join(poses_dir, "viewer_poses.json")
    assert os.path.exists(poses_file), f"Poses file not found: {poses_file}"
    
    poses = load_poses(poses_file)
    
    # Find the specific pose
    target_pose = None
    for pose_data in poses:
        if pose_data['pose_id'] == pose_id:
            target_pose = pose_data
            break
    
    assert target_pose is not None, f"Pose ID {pose_id} not found in poses file"
    
    # Extract camera parameters
    c2w_matrix = torch.tensor(target_pose["c2w_matrix"], device=device, dtype=torch.float32)
    K_matrix = torch.tensor(target_pose["K_matrix"], device=device, dtype=torch.float32)
    viewmat = c2w_matrix.inverse()
    
    # Use custom rendering resolution
    width = cfg.render_width
    height = cfg.render_height
    original_width = target_pose["width"]
    original_height = target_pose["height"]
    
    # Adjust intrinsics for the new resolution
    scale_x = width / original_width
    scale_y = height / original_height
    K_adjusted = K_matrix.clone()
    K_adjusted[0, 0] *= scale_x  # fx
    K_adjusted[1, 1] *= scale_y  # fy
    K_adjusted[0, 2] *= scale_x  # cx
    K_adjusted[1, 2] *= scale_y  # cy
    
    # Find merge candidates
    candidate_mask = find_merge_candidates(
        means, quats, scales, opacities, viewmat, K_adjusted, width, height,
        use_pixel_area=True, pixel_area_threshold=2.0, scale_modifier=1.0, eps2d=0.3, method="cuda"
    )
    
    candidate_indices = torch.where(candidate_mask)[0].cpu().numpy()
    candidate_means = means[candidate_mask]
    
    log.info(f"Loaded {len(candidate_indices)} candidates out of {means.shape[0]} Gaussians")
    
    return {
        'candidate_means': candidate_means,
        'candidate_indices': candidate_indices,
        'viewmat': viewmat,
        'K': K_adjusted,
        'width': width,
        'height': height
    }

def get_preprocessed_data(data: Dict, depth_threshold: float = 0.1, min_cluster_size: int = 2):
    """Get preprocessed data (same transformation used by both implementations)"""
    device = data['candidate_means'].device
    M = len(data['candidate_indices'])
    
    # 1. Transform to camera coordinates
    means_batch = data['candidate_means'].unsqueeze(0)  # [1, M, 3]
    viewmat_batch = data['viewmat'].unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
    
    dummy_covars = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, M, 3, 3)
    means_cam_batch, _ = world_to_cam(means_batch, dummy_covars, viewmat_batch)
    means_cam = means_cam_batch.squeeze(0).squeeze(0)  # [M, 3]
    
    # 2. Project to 2D pixel coordinates
    means_cam_proj = means_cam.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 3]
    K_batch = data['K'].unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    
    dummy_covars_cam = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, M, 3, 3)
    means2d_batch, _ = proj(means_cam_proj, dummy_covars_cam, K_batch, data['width'], data['height'])
    means2d = means2d_batch.squeeze(0).squeeze(0)  # [M, 2]
    
    # Convert to discrete pixel coordinates
    pixel_coords = torch.floor(means2d).int()  # [M, 2] - discrete coordinates
    
    # Filter out points outside image bounds
    valid_mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < data['width']) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < data['height']) &
        (means_cam[:, 2] > 0)  # Points must be in front of camera
    )
    
    if not valid_mask.any():
        return None, None, None
    
    # Apply valid mask
    valid_indices = torch.where(valid_mask)[0]
    valid_pixel_coords = pixel_coords[valid_mask]  # [V, 2]
    valid_means_cam = means_cam[valid_mask]  # [V, 3]
    valid_candidate_indices = data['candidate_indices'][valid_indices.cpu().numpy()]
    valid_candidate_indices_tensor = torch.from_numpy(valid_candidate_indices).to(device).int()
    
    return valid_means_cam, valid_pixel_coords, valid_candidate_indices_tensor

def verify_cluster_assignments_correctness(
    group_starts: np.ndarray,
    group_sizes: np.ndarray,
    sorted_pixel_hashes: np.ndarray,
    sorted_depths: np.ndarray,
    sorted_indices: np.ndarray,
    cluster_assignments: np.ndarray,
    num_groups: int,
    num_valid: int,
    total_clusters: int,
    depth_threshold: float = 0.1,
    min_cluster_size: int = 2
) -> bool:
    """Verify correctness properties of CUDA cluster assignments"""
    
    log.info(f"\n{'='*80}")
    log.info("VERIFYING CUDA DEPTH CLUSTERING CORRECTNESS (Stage 7)")
    log.info(f"{'='*80}")
    
    log.info(f"Input data:")
    log.info(f"  Pixel groups: {num_groups}")
    log.info(f"  Valid candidates: {num_valid}")
    log.info(f"  Total clusters assigned: {total_clusters}")
    log.info(f"  Depth threshold: {depth_threshold}")
    log.info(f"  Min cluster size: {min_cluster_size}")
    
    all_checks_pass = True
    
    # Check 1: Cluster assignments are in valid range
    log.info(f"\n--- Check 1: Cluster Assignment Validity ---")
    clustered_mask = cluster_assignments >= 0
    clustered_count = np.sum(clustered_mask)
    
    if clustered_count > 0:
        max_cluster_id = np.max(cluster_assignments[clustered_mask])
        min_cluster_id = np.min(cluster_assignments[clustered_mask])
        
        if min_cluster_id < 0:
            log.error(f"❌ Invalid cluster IDs found: min={min_cluster_id}")
            all_checks_pass = False
        elif max_cluster_id >= total_clusters:
            log.error(f"❌ Cluster IDs exceed total_clusters: max={max_cluster_id}, total={total_clusters}")
            all_checks_pass = False
        else:
            log.info(f"✅ Cluster IDs valid: range=[{min_cluster_id}, {max_cluster_id}], total={total_clusters}")
    else:
        log.info(f"✅ No clustered candidates (valid for difficult data)")
    
    # Check 2: Each Gaussian appears in at most one cluster
    log.info(f"\n--- Check 2: No Duplicate Assignments ---")
    clustered_indices = sorted_indices[clustered_mask]
    unique_indices = np.unique(clustered_indices)
    
    if len(clustered_indices) == len(unique_indices):
        log.info(f"✅ No duplicate assignments: {len(clustered_indices)} unique clustered candidates")
    else:
        log.error(f"❌ Duplicate assignments found: {len(clustered_indices)} assigned, {len(unique_indices)} unique")
        all_checks_pass = False
    
    # Check 3: All clusters meet minimum size requirement
    log.info(f"\n--- Check 3: Cluster Size Requirements ---")
    if total_clusters > 0:
        cluster_sizes = np.bincount(cluster_assignments[clustered_mask], minlength=total_clusters)
        min_size = np.min(cluster_sizes)
        max_size = np.max(cluster_sizes)
        mean_size = np.mean(cluster_sizes)
        
        if min_size >= min_cluster_size:
            log.info(f"✅ All clusters meet size requirement: min={min_size}, max={max_size}, mean={mean_size:.1f}")
        else:
            small_clusters = np.sum(cluster_sizes < min_cluster_size)
            log.error(f"❌ {small_clusters} clusters below minimum size: min_size={min_size}, required={min_cluster_size}")
            all_checks_pass = False
    else:
        log.info(f"✅ No clusters to check (valid for difficult data)")
    
    # Check 4: Depth coherence within clusters (consecutive pairs - matches algorithm)
    log.info(f"\n--- Check 4: Depth Coherence Within Clusters ---")
    if total_clusters > 0:
        violation_count = 0
        max_violation = 0.0
        
        for cluster_id in range(total_clusters):
            cluster_mask = cluster_assignments == cluster_id
            if not np.any(cluster_mask):
                continue
                
            cluster_depths = sorted_depths[cluster_mask]
            if len(cluster_depths) <= 1:
                continue
                
            # FIXED: Check consecutive depth differences (matches PyTorch/CUDA algorithm)
            consecutive_diffs = np.abs(cluster_depths[1:] - cluster_depths[:-1])
            max_consecutive_diff = np.max(consecutive_diffs) if len(consecutive_diffs) > 0 else 0.0
            
            if max_consecutive_diff > depth_threshold:
                violation_count += 1
                max_violation = max(max_violation, max_consecutive_diff)
        
        if violation_count == 0:
            log.info(f"✅ All clusters satisfy depth coherence (consecutive pairs ≤ {depth_threshold})")
        else:
            log.error(f"❌ {violation_count} clusters violate depth coherence: max_violation={max_violation:.6f}")
            all_checks_pass = False
    else:
        log.info(f"✅ No clusters to check depth coherence")
    
    # Check 5: Pixel group consistency
    log.info(f"\n--- Check 5: Pixel Group Consistency ---")
    pixel_group_violations = 0
    
    for group_idx in range(num_groups):
        start = group_starts[group_idx]
        size = group_sizes[group_idx]
        
        if size <= 0:
            continue
            
        # Get pixel coordinate for this group
        group_hash = sorted_pixel_hashes[start]
        px = int(group_hash >> 32)
        py = int(group_hash & 0xFFFFFFFF)
        
        # Check all candidates in this pixel group have the same pixel coordinates
        for i in range(size):
            candidate_hash = sorted_pixel_hashes[start + i]
            candidate_px = int(candidate_hash >> 32)
            candidate_py = int(candidate_hash & 0xFFFFFFFF)
            
            if candidate_px != px or candidate_py != py:
                pixel_group_violations += 1
                break
    
    if pixel_group_violations == 0:
        log.info(f"✅ All pixel groups have consistent coordinates")
    else:
        log.error(f"❌ {pixel_group_violations} pixel groups have inconsistent coordinates")
        all_checks_pass = False
    
    # Summary
    log.info(f"\n--- CORRECTNESS SUMMARY ---")
    if all_checks_pass:
        log.info(f"🎉 ALL CORRECTNESS CHECKS PASSED!")
        log.info(f"  Stage 7 (depth clustering) is working correctly")
    else:
        log.error(f"💥 SOME CORRECTNESS CHECKS FAILED!")
        log.error(f"  Stage 7 (depth clustering) has bugs to fix")
    
    log.info(f"{'='*80}")
    
    return all_checks_pass


def verify_step8_result_processing(
    cuda_clusters: List[np.ndarray], 
    cluster_assignments: np.ndarray,
    sorted_indices: np.ndarray, 
    total_clusters: int,
    num_valid: int
) -> bool:
    """
    Verify Step 8 (result processing) correctness.
    
    Checks that the final processed clusters from Step 8 correctly correspond
    to the cluster assignments from Step 7.
    """
    all_checks_pass = True
    
    # Check 1: Cluster count consistency
    log.info(f"\n--- Check 1: Cluster Count Consistency ---")
    if len(cuda_clusters) == total_clusters:
        log.info(f"✅ Cluster count matches: {len(cuda_clusters)} clusters")
    else:
        log.error(f"❌ Cluster count mismatch: expected {total_clusters}, got {len(cuda_clusters)}")
        all_checks_pass = False
    
    # Check 2: Build expected clusters from step 7 assignments (MANUAL POSTPROCESSING)
    log.info(f"\n--- Check 2: Step 7 to Step 8 Consistency ---")
    log.info(f"  DEBUG: Manual postprocessing simulation...")
    
    expected_clusters = {}
    clustered_count_expected = 0
    
    for i in range(num_valid):
        cluster_id = cluster_assignments[i]
        if cluster_id >= 0:  # Valid cluster assignment
            if cluster_id not in expected_clusters:
                expected_clusters[cluster_id] = []
            expected_clusters[cluster_id].append(sorted_indices[i])
            clustered_count_expected += 1
    
    log.info(f"  Manual postprocessing results:")
    log.info(f"    Expected clusters from Step 7: {len(expected_clusters)}")
    log.info(f"    Expected clustered indices: {clustered_count_expected}")
    
    # DEBUG: Show first few manual clusters  
    manual_cluster_list = list(expected_clusters.values())
    log.info(f"    First 3 manual clusters:")
    for i in range(min(3, len(manual_cluster_list))):
        cluster = manual_cluster_list[i]
        log.info(f"      Manual Cluster {i} (size {len(cluster)}): {sorted(cluster)[:5]}...")
    
    # DEBUG: Show CUDA clusters for comparison
    log.info(f"  CUDA Stage 8 results:")
    log.info(f"    First 3 CUDA clusters:")
    for i in range(min(3, len(cuda_clusters))):
        cluster = cuda_clusters[i]
        log.info(f"      CUDA Cluster {i} (size {len(cluster)}): {sorted(cluster)[:5]}...")
    
    # Check 3: Compare actual vs expected clusters
    actual_clustered_indices = set()
    for cluster in cuda_clusters:
        actual_clustered_indices.update(cluster)
    
    expected_clustered_indices = set()
    for cluster_indices in expected_clusters.values():
        expected_clustered_indices.update(cluster_indices)
    
    if actual_clustered_indices == expected_clustered_indices:
        log.info(f"✅ Clustered indices match perfectly: {len(actual_clustered_indices)} indices")
    else:
        log.error(f"❌ Clustered indices mismatch!")
        missing = expected_clustered_indices - actual_clustered_indices
        extra = actual_clustered_indices - expected_clustered_indices
        if missing:
            log.error(f"  Missing indices: {len(missing)}, sample: {sorted(list(missing))[:5]}")
        if extra:
            log.error(f"  Extra indices: {len(extra)}, sample: {sorted(list(extra))[:5]}")
        all_checks_pass = False
    
    # Check 4: Individual cluster content verification (sample check)
    log.info(f"\n--- Check 4: Individual Cluster Content (Sample) ---")
    sample_size = min(5, len(cuda_clusters), len(expected_clusters))
    matches = 0
    
    # Convert expected clusters to list of sets for comparison
    expected_cluster_sets = [set(cluster_indices) for cluster_indices in expected_clusters.values()]
    cuda_cluster_sets = [set(cluster) for cluster in cuda_clusters]
    
    # Check if each CUDA cluster matches any expected cluster
    for i, cuda_set in enumerate(cuda_cluster_sets[:sample_size]):
        found_match = False
        for expected_set in expected_cluster_sets:
            if cuda_set == expected_set:
                found_match = True
                matches += 1
                break
        
        if found_match:
            log.info(f"  Cluster {i}: ✅ MATCH (size: {len(cuda_set)})")
        else:
            log.info(f"  Cluster {i}: ❌ NO MATCH (size: {len(cuda_set)})")
            all_checks_pass = False  # This should now pass with consistent preprocessing
    
    log.info(f"  Sample cluster matches: {matches}/{sample_size}")
    
    if matches == sample_size:
        log.info(f"✅ All sampled clusters match perfectly")
    else:
        log.error(f"❌ Cluster content mismatch: {matches}/{sample_size} matches")
        all_checks_pass = False
    
    # Check 5: No duplicate indices across clusters  
    log.info(f"\n--- Check 5: No Duplicate Indices Across Clusters ---")
    all_indices = []
    for cluster in cuda_clusters:
        all_indices.extend(cluster)
    
    if len(all_indices) == len(set(all_indices)):
        log.info(f"✅ No duplicate indices found: {len(all_indices)} unique indices")
    else:
        duplicates = len(all_indices) - len(set(all_indices))
        log.error(f"❌ Found {duplicates} duplicate indices across clusters")
        all_checks_pass = False
    
    # Check 6: Cluster size distribution consistency
    log.info(f"\n--- Check 6: Cluster Size Distribution ---")
    actual_sizes = [len(cluster) for cluster in cuda_clusters]
    expected_sizes = [len(cluster_indices) for cluster_indices in expected_clusters.values()]
    
    actual_sizes.sort()
    expected_sizes.sort()
    
    if actual_sizes == expected_sizes:
        log.info(f"✅ Cluster size distributions match perfectly")
        log.info(f"  Size range: {min(actual_sizes)} to {max(actual_sizes)}")
    else:
        log.error(f"❌ Cluster size distributions differ")
        log.error(f"  Actual sizes: {actual_sizes[:10]}{'...' if len(actual_sizes) > 10 else ''}")
        log.error(f"  Expected sizes: {expected_sizes[:10]}{'...' if len(expected_sizes) > 10 else ''}")
        all_checks_pass = False
    
    return all_checks_pass

def main():
    """Main verification function"""
    log.info("Verifying CUDA depth clustering correctness (Stages 7-8)...")
    log.info("="*80)
    
    # Load test data
    data = load_test_data(pose_id=80)
    depth_threshold = 0.1
    min_cluster_size = 2
    
    # Get preprocessed data
    valid_means_cam, valid_pixel_coords, valid_candidate_indices_tensor = get_preprocessed_data(
        data, depth_threshold, min_cluster_size
    )
    
    if valid_means_cam is None:
        log.error("No valid candidates found for testing")
        return
    
    log.info(f"Test parameters:")
    log.info(f"  Valid candidates: {len(valid_candidate_indices_tensor)}")
    log.info(f"  Depth threshold: {depth_threshold}")
    log.info(f"  Min cluster size: {min_cluster_size}")
    
    # PIPELINE 1: Full CUDA clustering (Steps 1-8) with RAW data
    log.info(f"\nRunning FULL CUDA clustering pipeline (Steps 1-8)...")
    from stream.clustering_cuda import cluster_center_in_pixel_cuda
    
    cuda_clusters = cluster_center_in_pixel_cuda(
        data['candidate_means'], data['candidate_indices'],
        data['viewmat'], data['K'], data['width'], data['height'], 
        depth_threshold, min_cluster_size
    )
    
    # PIPELINE 2: Partial CUDA (Steps 1-7) + Manual postprocessing with SAME RAW data
    log.info(f"\nRunning PARTIAL CUDA clustering (Steps 1-7) + manual postprocessing...")
    
    # First get preprocessed data that matches what the full CUDA pipeline will process internally
    valid_means_cam_raw, valid_pixel_coords_raw, valid_candidate_indices_tensor_raw = get_preprocessed_data(
        data, depth_threshold, min_cluster_size
    )
    
    cuda_step7_result = extract_cluster_assignments_step7_cuda(
        valid_means_cam_raw, valid_pixel_coords_raw, valid_candidate_indices_tensor_raw,
        data['viewmat'], data['K'], data['width'], data['height'], 
        depth_threshold, min_cluster_size
    )
    
    log.info(f"Full CUDA pipeline clusters: {len(cuda_clusters)}")
    log.info(f"Partial CUDA pipeline clusters: {cuda_step7_result['total_clusters']}")
    
    # Now compare: Full CUDA (Steps 1-8) vs Partial CUDA (Steps 1-7) + Manual Step 8
    # This tests if Stage 8 CUDA implementation matches manual postprocessing
    
    # Convert to numpy for verification  
    group_starts = cuda_step7_result['group_starts'].cpu().numpy()
    group_sizes = cuda_step7_result['group_sizes'].cpu().numpy()
    sorted_pixel_hashes = cuda_step7_result['sorted_pixel_hashes'].cpu().numpy()
    sorted_depths = cuda_step7_result['sorted_depths'].cpu().numpy()
    sorted_indices = cuda_step7_result['sorted_indices'].cpu().numpy()
    cluster_assignments = cuda_step7_result['cluster_assignments'].cpu().numpy()
    num_groups = cuda_step7_result['num_groups']
    num_valid = cuda_step7_result['num_valid']
    total_clusters = cuda_step7_result['total_clusters']
    
    log.info(f"Data consistency check:")
    log.info(f"  Full pipeline clustered: {sum(len(cluster) for cluster in cuda_clusters)}")
    log.info(f"  Partial pipeline clustered: {np.sum(cluster_assignments >= 0)}")
    log.info(f"  Expected to be equal if preprocessing is consistent")
    
    # Verify Stage 7 correctness
    stage7_success = verify_cluster_assignments_correctness(
        group_starts, group_sizes, sorted_pixel_hashes, sorted_depths, sorted_indices,
        cluster_assignments, num_groups, num_valid, total_clusters, depth_threshold, min_cluster_size
    )
    
    # Verify Stage 8 correctness (result processing)
    log.info(f"\n--- STEP 8 VERIFICATION: Result Processing ---")
    stage8_success = verify_step8_result_processing(
        cuda_clusters, cluster_assignments, sorted_indices, total_clusters, num_valid
    )
    
    # Final summary
    log.info(f"\n{'='*80}")
    log.info("FINAL VERIFICATION SUMMARY")
    log.info(f"{'='*80}")
    
    if stage7_success and stage8_success:
        log.info(f"🎉 COMPLETE VERIFICATION SUCCESSFUL!")
        log.info(f"✅ Stage 7 (Depth Clustering): All correctness properties satisfied")
        log.info(f"✅ Stage 8 (Result Processing): All computations correct")
        log.info(f"  CUDA implementation is working correctly!")
    else:
        log.error(f"💥 VERIFICATION FAILED!")
        if not stage7_success:
            log.error(f"❌ Stage 7 (Depth Clustering): Correctness violations found")
        if not stage8_success:
            log.error(f"❌ Stage 8 (Result Processing): Logic errors found")
        log.error(f"  CUDA implementation needs debugging!")
    
    log.info(f"{'='*80}")

if __name__ == "__main__":
    main()
