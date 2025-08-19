#!/usr/bin/env python3
"""
Evaluate the quantitative quality (PSNR and SSIM) of culling methods against ground truth.
This script reads saved viewer poses and renders images with different culling configurations.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# Add parent directories to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gsplat.rendering import rasterization
from stream.culling import frustum_culling, distance_culling

from examples.config import Config, load_config_from_toml, merge_config
from examples.utils import get_color, Metrics
from scripts.utils import set_result_dir
from scripts.ablation.utils import load_checkpoint, load_poses

ALPHA_MASK_THRESHOLD = 0.5

def render_image(
    means: torch.Tensor,
    quats: torch.Tensor, 
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    pose_data: Dict,
    device: torch.device,
    cfg: Config,
    apply_frustum_culling: bool = False,
    apply_distance_culling: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, int, int, float]: # [1, H, W, 3], [1, H, W, 1], int, int, float
    """Render an image for a given pose with optional culling.
    Args:
        means: [N, 3]
        quats: [N, 4]
        scales: [N]
        opacities: [N]
        colors: [N, K, 3]
        pose_data: Dict
        device: torch.device
        cfg: Config
        apply_frustum_culling: bool
        apply_distance_culling: bool
    Returns:
        rendered_image: [1, H, W, 3]
        rendered_alpha: [1, H, W, 1]
        num_original: int - Number of Gaussians before culling
        num_after_culling: int - Number of Gaussians after culling
        culling_ratio: float - Percentage of Gaussians culled
        pixel_coverage: float - pixel coverage of each Gaussian [N]
    """
    
    # Extract pose information
    c2w_matrix = torch.tensor(pose_data["c2w_matrix"], device=device, dtype=torch.float32)
    K_matrix_original = torch.tensor(pose_data["K_matrix"], device=device, dtype=torch.float32)
    original_width = pose_data["width"]
    original_height = pose_data["height"]
    near_plane = pose_data["near_plane"]
    far_plane = pose_data["far_plane"]
    max_sh_degree = pose_data["max_sh_degree"]
    
    # Use custom rendering resolution from config
    width = cfg.render_width
    height = cfg.render_height
    
    # Adjust intrinsics for the new resolution
    scale_x = width / original_width
    scale_y = height / original_height
    K_matrix = K_matrix_original.clone()
    K_matrix[0, 0] *= scale_x  # fx
    K_matrix[1, 1] *= scale_y  # fy
    K_matrix[0, 2] *= scale_x  # cx
    K_matrix[1, 2] *= scale_y  # cy
    
    viewmat = c2w_matrix.inverse()
    
    # Start with all Gaussians
    render_means = means
    render_quats = quats
    render_scales = scales
    render_opacities = opacities
    render_colors = colors
    num_original = render_means.shape[0]
    
    # Apply frustum culling if requested
    if apply_frustum_culling:
        culling_mask = frustum_culling(
            render_means, render_scales, viewmat, K_matrix, width, height,
            near=near_plane, far=far_plane
        )
        render_means = render_means[culling_mask]
        render_quats = render_quats[culling_mask]
        render_scales = render_scales[culling_mask]
        render_opacities = render_opacities[culling_mask]
        render_colors = render_colors[culling_mask]
    
    # Apply distance culling if requested
    pixel_coverage = None
    if apply_distance_culling:
        culling_mask, pixel_coverage = distance_culling(
            render_means, render_quats, render_scales, render_opacities,
            viewmat, K_matrix, width, height, 
            pixel_threshold=cfg.pixel_threshold, return_pixel_sizes=True, 
            dump_file=None, use_timestamps=False
        )
        render_means = render_means[culling_mask]
        render_quats = render_quats[culling_mask]
        render_scales = render_scales[culling_mask]
        render_opacities = render_opacities[culling_mask]
        render_colors = render_colors[culling_mask]
    
    # Handle case where all Gaussians are culled
    if render_means.shape[0] == 0:
        print(f"Warning: All Gaussians culled, returning background image")
        background_color = torch.tensor(get_color("white"), dtype=torch.float32, device=device) / 255.0
        background_image = background_color.view(1, 1, 1, 3).expand(1, height, width, 3)
        background_alpha = torch.ones((1, height, width, 1), dtype=torch.float32, device=device)
        num_after_culling = 0
        culling_ratio = 100.0
        return background_image, background_alpha, num_original, num_after_culling, culling_ratio, pixel_coverage
    
    # Compute SH degree
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    effective_sh_degree = min(max_sh_degree, sh_degree) if sh_degree is not None else None
    
    # Rasterize
    render_colors_out, render_alphas_out, _ = rasterization(
        render_means,
        render_quats,
        render_scales,
        render_opacities,
        render_colors,
        viewmat[None],
        K_matrix[None],
        width,
        height,
        sh_degree=effective_sh_degree,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=0.0,  # Default value
        eps2d=0.3,  # Default value
        backgrounds=torch.tensor([get_color("white")], device=device) / 255.0,
        render_mode="RGB",
        rasterize_mode="classic",
        camera_model="pinhole",
        packed=False,
        with_ut=cfg.with_ut,
        with_eval3d=cfg.with_eval3d,
    )
    
    # Keep as torch tensor [1, H, W, 3] and clamp to [0, 1]
    rendered_image = torch.clamp(render_colors_out, 0.0, 1.0)
    
    # Alpha channel is already in range [0, 1], shape [1, H, W, 1]
    rendered_alpha = render_alphas_out
    
    num_after_culling = render_means.shape[0]
    culling_ratio = (num_original - num_after_culling) / num_original * 100.0
    
    return rendered_image, rendered_alpha, num_original, num_after_culling, culling_ratio, pixel_coverage

def evaluate_culling_quality(frame_id, iter, cfg: Config, exp_name, sub_exp_name=None, save_images=True, measure_lpips=False):
    """Main evaluation function."""

    cfg.data_dir = os.path.join(cfg.actorshq_data_dir, f"{frame_id}", f"resolution_{cfg.resolution}")
    cfg.exp_name = exp_name
    if sub_exp_name is not None:
        cfg.exp_name = f"{exp_name}_{sub_exp_name}"
    cfg.run_mode = "render"
    cfg.scene_id = frame_id
    set_result_dir(cfg, exp_name=exp_name, sub_exp_name=sub_exp_name)
    ckpt = os.path.join(f"{cfg.result_dir}/ckpts/ckpt_{iter - 1}_rank0.pt")
    cfg.ckpt = ckpt

    print(f"  Checkpoint: {cfg.ckpt}")
    print(f"  Result dir: {cfg.result_dir}")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize metrics calculator
    metrics_calculator = Metrics(device=device, lpips_net=cfg.lpips_net, data_range=1.0)
    
    # Load checkpoint
    if cfg.ckpt is None:
        raise ValueError("Checkpoint path (--ckpt) is required")
    
    means, quats, scales, opacities, colors = load_checkpoint(cfg.ckpt, device)
    
    # Load poses
    poses_dir = os.path.join(cfg.result_dir, "viewer_poses")
    poses_file = os.path.join(poses_dir, "viewer_poses.json")
    poses = load_poses(poses_file)
    
    print(f"Using custom rendering resolution: {cfg.render_width}x{cfg.render_height}")
    print(f"Original pose resolution will be scaled accordingly")
    
    if len(poses) == 0:
        raise ValueError("No poses found for evaluation")
    
    # Evaluation configurations
    eval_configs = [
        {"name": "frustum_only", "frustum": True, "distance": False},
        {"name": f"distance_only_px_{cfg.pixel_threshold}", "frustum": False, "distance": True},
        {"name": f"both_culling_px_{cfg.pixel_threshold}", "frustum": True, "distance": True},
    ]
    
    # Results storage - using pose_id as keys for easy lookup
    results = {config["name"]: {} for config in eval_configs}
    
    # Create image output directories (only if saving images)
    images_output_dir = None
    culled_images_dirs = {}
    masks_output_dir = None
    
    if save_images:
        images_output_dir = os.path.join(poses_dir, f"render_hxw_{cfg.render_width}x{cfg.render_height}", "evaluation_images")
        os.makedirs(images_output_dir, exist_ok=True)
        
        # Create subdirectories for each configuration (stacked GT|Culled images)
        for config in eval_configs:
            config_dir = os.path.join(images_output_dir, config["name"])
            os.makedirs(config_dir, exist_ok=True)
            culled_images_dirs[config["name"]] = config_dir
        
        # Create directory for ground truth masks
        masks_output_dir = os.path.join(images_output_dir, "gt_masks")
        os.makedirs(masks_output_dir, exist_ok=True)
    
    print(f"\nStarting evaluation on {len(poses)} poses, distance threshold: {cfg.pixel_threshold} px ...")
    start_time = time.time()
    
    for i, pose_data in enumerate(poses):
        print(f"\nEvaluating pose {i+1}/{len(poses)} (pose_id: {pose_data['pose_id']})")
        
        # Get pose_id and distance for consistent naming and result mapping
        pose_id = pose_data['pose_id']
        distance_to_bbox = pose_data['distance_to_bbox_center']
        
        # Render ground truth (no culling)
        gt_image, gt_alpha, wo_culled_count, _, _, _ = render_image(
            means, quats, scales, opacities, colors, pose_data, device, cfg,
            apply_frustum_culling=False, apply_distance_culling=False
        ) # [1, H, W, 3], [1, H, W, 1]
        
        # Convert GT alpha to binary mask 
        # Use alpha > 0.5 threshold for binary mask
        gt_alpha = gt_alpha.squeeze(-1) # [1, H, W]
        gt_mask_binary = (gt_alpha > ALPHA_MASK_THRESHOLD).bool()  # [1, H, W]
        
        # Convert ground truth to numpy for image stacking (only if saving images)
        if save_images:
            gt_image_np = (gt_image[0].cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]
            # gt_mask_binary_np = (gt_mask_binary[0, :, :].cpu().numpy() * 255.0).astype(np.uint8)  # [H, W]
            # mask_path = os.path.join(masks_output_dir, f"pose_{pose_id:06d}_mask.png")
            # imageio.imwrite(mask_path, gt_mask_binary_np)
        
        # Evaluate each culling configuration
        for config in eval_configs:
            culled_image, _, _, num_after_culling, culling_ratio, pixel_coverage = render_image(
                means, quats, scales, opacities, colors, pose_data, device, cfg,
                apply_frustum_culling=config["frustum"],
                apply_distance_culling=config["distance"],
            ) # [1, H, W, 3], [1, H, W, 1]
            
            # Save stacked comparison image (only if saving images)
            if save_images and config["name"] == f"distance_only_px_{cfg.pixel_threshold}":
                # Convert culled image to numpy and stack with GT horizontally
                culled_image_np = (culled_image[0].cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]
                
                # Calculate difference map for visualization
                gt_image_float = gt_image[0].cpu().numpy()  # [H, W, 3] in [0, 1]
                culled_image_float = culled_image[0].cpu().numpy()  # [H, W, 3] in [0, 1]
                
                # Absolute difference
                diff_abs = np.abs(gt_image_float - culled_image_float)  # [H, W, 3]
                
                # # Convert to visualization (amplify differences for visibility)
                # diff_vis = np.clip(diff_abs * 10.0, 0.0, 1.0)  # Amplify by 5x for better visibility
                # diff_vis_np = (diff_vis * 255).astype(np.uint8)  # [H, W, 3]
                
                # Create error heatmap (convert to grayscale then to heatmap)
                diff_gray = np.mean(diff_abs, axis=2)  # [H, W] average across RGB
                diff_heatmap = plt.cm.hot(np.clip(diff_gray * 10.0, 0.0, 1.0))[:, :, :3]  # Apply hot colormap
                diff_heatmap_np = (diff_heatmap * 255).astype(np.uint8)  # [H, W, 3]

                # Stack GT | Culled | Heatmap horizontally
                stacked_image = np.concatenate([gt_image_np, culled_image_np, diff_heatmap_np], axis=1)  # [H, 4*W, 3]
                
                # # Stack GT | Culled | Difference | Heatmap horizontally
                # stacked_image = np.concatenate([gt_image_np, culled_image_np, diff_vis_np, diff_heatmap_np], axis=1)  # [H, 4*W, 3]
                
                # Save stacked comparison image
                stacked_image_path = os.path.join(culled_images_dirs[config["name"]], f"pose_{pose_id:06d}.png")
                imageio.imwrite(stacked_image_path, stacked_image)
            
            # Calculate all metrics using the Metrics class
            metric_results = metrics_calculator.calculate_all_metrics(
                culled_image, gt_image, gt_mask_binary, measure_lpips=measure_lpips
            )
            
            # Store results using pose_id as key
            results[config["name"]][pose_id] = {
                **metric_results,  # Include all metric results
                "distance_to_bbox": distance_to_bbox,
                "wo_culled_count": wo_culled_count,
                "culled_count": num_after_culling,
                "culling_ratio": culling_ratio,
                "pixel_coverage": pixel_coverage,
            }
            
            # Print formatted metrics
            lpips_reliability = "✓" if metric_results['lpips_reliable'] else "⚠"
            print(f"  {config['name']}: PSNR={metric_results['psnr']:.2f}, SSIM={metric_results['ssim']:.4f}, LPIPS={metric_results['lpips']:.3f}, " 
            f"MaskedPSNR={metric_results['masked_psnr']:.2f}, MaskedSSIM={metric_results['masked_ssim']:.4f}, MaskedSE={metric_results['masked_se']:.2f}, MaskedRMSE={metric_results['masked_rmse']:.4f}, "
            f"CroppedLPIPS={metric_results['cropped_lpips']:.3f}{lpips_reliability}, "
            f"Culled={culling_ratio:.1f}%, Coverage={metric_results['mask_coverage']:.1%}, Dist={distance_to_bbox:.3f}m, Valid={metric_results['num_valid_pixels']:.0f}")
        
        # Progress update
        if (i + 1) % 10 == 0 or i == len(poses) - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(poses) - i - 1)
            print(f"Progress: {i+1}/{len(poses)} ({100*(i+1)/len(poses):.1f}%) - ETA: {eta:.1f}s")
    
    # Calculate and save summary statistics
    pixel_sizes_dir = os.path.join(poses_dir, f"render_hxw_{cfg.render_width}x{cfg.render_height}", "pixel_sizes")
    os.makedirs(pixel_sizes_dir, exist_ok=True)
    for config_name, pose_results in results.items():
        if config_name == f"distance_only_px_{cfg.pixel_threshold}":
            for pose_id, pose_data in pose_results.items():
                # Pixel sizes are same across any pixel_threshold, so we overwrite the file
                pixel_coverage = pose_data["pixel_coverage"]
                if pixel_coverage is not None:
                    np.save(os.path.join(pixel_sizes_dir, f"{pose_id}.npy"), pixel_coverage)
                    print(f"Saved pixel sizes for pose {pose_id} to {os.path.join(pixel_sizes_dir, f'{pose_id}.npy')}")

        # remove pixel_sizes from pose_data
        for pose_id in pose_results:
            del pose_results[pose_id]["pixel_coverage"]

    summary = {}
    for config_name, pose_results in results.items():
        # Extract values from pose_id-keyed dictionary
        psnr_vals = [pose_data["psnr"] for pose_data in pose_results.values()]
        ssim_vals = [pose_data["ssim"] for pose_data in pose_results.values()]
        lpips_vals = [pose_data["lpips"] for pose_data in pose_results.values()]
        masked_psnr_vals = [pose_data["masked_psnr"] for pose_data in pose_results.values()]
        masked_ssim_vals = [pose_data["masked_ssim"] for pose_data in pose_results.values()]
        masked_se_vals = [pose_data["masked_se"] for pose_data in pose_results.values()]
        masked_rmse_vals = [pose_data["masked_rmse"] for pose_data in pose_results.values()]
        
        # Separate reliable and unreliable cropped LPIPS values
        all_cropped_lpips = [pose_data["cropped_lpips"] for pose_data in pose_results.values()]
        reliable_cropped_lpips = [pose_data["cropped_lpips"] for pose_data in pose_results.values() if pose_data["lpips_reliable"]]
        mask_coverages = [pose_data["mask_coverage"] for pose_data in pose_results.values()]
        reliability_count = sum(1 for pose_data in pose_results.values() if pose_data["lpips_reliable"])
        wo_culled_counts = [pose_data["wo_culled_count"] for pose_data in pose_results.values()]
        culled_counts = [pose_data["culled_count"] for pose_data in pose_results.values()]
        culling_ratios = [pose_data["culling_ratio"] for pose_data in pose_results.values()]
        distances = [pose_data["distance_to_bbox"] for pose_data in pose_results.values()]
        valid_pixels = [pose_data["num_valid_pixels"] for pose_data in pose_results.values()]
        
        summary[config_name] = {
            "psnr_mean": float(np.mean(psnr_vals)),
            "psnr_std": float(np.std(psnr_vals)),
            "psnr_min": float(np.min(psnr_vals)),
            "psnr_max": float(np.max(psnr_vals)),
            "ssim_mean": float(np.mean(ssim_vals)),
            "ssim_std": float(np.std(ssim_vals)),
            "ssim_min": float(np.min(ssim_vals)),
            "ssim_max": float(np.max(ssim_vals)),
            "lpips_mean": float(np.mean(lpips_vals)),
            "lpips_std": float(np.std(lpips_vals)),
            "lpips_min": float(np.min(lpips_vals)),
            "lpips_max": float(np.max(lpips_vals)),
            "masked_psnr_mean": float(np.mean(masked_psnr_vals)),
            "masked_psnr_std": float(np.std(masked_psnr_vals)),
            "masked_psnr_min": float(np.min(masked_psnr_vals)),
            "masked_psnr_max": float(np.max(masked_psnr_vals)),
            "masked_ssim_mean": float(np.mean(masked_ssim_vals)),
            "masked_ssim_std": float(np.std(masked_ssim_vals)),
            "masked_ssim_min": float(np.min(masked_ssim_vals)),
            "masked_ssim_max": float(np.max(masked_ssim_vals)),
            "masked_se_mean": float(np.mean(masked_se_vals)),
            "masked_se_std": float(np.std(masked_se_vals)),
            "masked_se_min": float(np.min(masked_se_vals)),
            "masked_se_max": float(np.max(masked_se_vals)),
            "masked_rmse_mean": float(np.mean(masked_rmse_vals)),
            "masked_rmse_std": float(np.std(masked_rmse_vals)),
            "masked_rmse_min": float(np.min(masked_rmse_vals)),
            "masked_rmse_max": float(np.max(masked_rmse_vals)),
            "cropped_lpips_mean": float(np.mean(all_cropped_lpips)),
            "cropped_lpips_std": float(np.std(all_cropped_lpips)),
            "cropped_lpips_min": float(np.min(all_cropped_lpips)),
            "cropped_lpips_max": float(np.max(all_cropped_lpips)),
            # Additional statistics for reliable cropped LPIPS
            "reliable_cropped_lpips_count": reliability_count,
            "reliable_cropped_lpips_ratio": reliability_count / len(pose_results) if len(pose_results) > 0 else 0.0,
            "reliable_cropped_lpips_mean": float(np.mean(reliable_cropped_lpips)) if reliable_cropped_lpips else float('nan'),
            "reliable_cropped_lpips_std": float(np.std(reliable_cropped_lpips)) if reliable_cropped_lpips else float('nan'),
            "mask_coverage_mean": float(np.mean(mask_coverages)),
            "mask_coverage_std": float(np.std(mask_coverages)),
            "mask_coverage_min": float(np.min(mask_coverages)),
            "mask_coverage_max": float(np.max(mask_coverages)),
            "wo_culled_count_mean": float(np.mean(wo_culled_counts)),
            "wo_culled_count_std": float(np.std(wo_culled_counts)),
            "wo_culled_count_min": float(np.min(wo_culled_counts)),
            "wo_culled_count_max": float(np.max(wo_culled_counts)),
            "culled_count_mean": float(np.mean(culled_counts)),
            "culled_count_std": float(np.std(culled_counts)),
            "culled_count_min": float(np.min(culled_counts)),
            "culled_count_max": float(np.max(culled_counts)),
            "culling_ratio_mean": float(np.mean(culling_ratios)),
            "culling_ratio_std": float(np.std(culling_ratios)),
            "distance_mean": float(np.mean(distances)),
            "distance_std": float(np.std(distances)),
            "distance_min": float(np.min(distances)),
            "distance_max": float(np.max(distances)),
            "valid_pixels_mean": float(np.mean(valid_pixels)),
            "valid_pixels_std": float(np.std(valid_pixels)),
            "valid_pixels_min": int(np.min(valid_pixels)),
            "valid_pixels_max": int(np.max(valid_pixels)),
            "num_poses": len(psnr_vals)
        }
    
    # Save detailed results
    output_dir = os.path.join(poses_dir, f"render_hxw_{cfg.render_width}x{cfg.render_height}", "culling_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results (pose_id-keyed format for easy lookup)
    detailed_file = os.path.join(output_dir, f"detailed_results_px_{cfg.pixel_threshold}.json")
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary_file = os.path.join(output_dir, f"summary_px_{cfg.pixel_threshold}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("CULLING QUALITY EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Evaluated {len(poses)} poses")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print()
    
    for config_name, stats in summary.items():
        print(f"{config_name.upper().replace('_', ' ')}:")
        print(f"  PSNR: {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f} dB (range: {stats['psnr_min']:.2f} - {stats['psnr_max']:.2f})")
        print(f"  SSIM: {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f} (range: {stats['ssim_min']:.4f} - {stats['ssim_max']:.4f})")
        print(f"  LPIPS: {stats['lpips_mean']:.3f} ± {stats['lpips_std']:.3f} (range: {stats['lpips_min']:.3f} - {stats['lpips_max']:.3f})")
        print(f"  Masked PSNR: {stats['masked_psnr_mean']:.2f} ± {stats['masked_psnr_std']:.2f} dB (range: {stats['masked_psnr_min']:.2f} - {stats['masked_psnr_max']:.2f})")
        print(f"  Masked SSIM: {stats['masked_ssim_mean']:.4f} ± {stats['masked_ssim_std']:.4f} (range: {stats['masked_ssim_min']:.4f} - {stats['masked_ssim_max']:.4f})")
        print(f"  Masked SE: {stats['masked_se_mean']:.2f} ± {stats['masked_se_std']:.2f} (range: {stats['masked_se_min']:.2f} - {stats['masked_se_max']:.2f})")
        print(f"  Masked RMSE: {stats['masked_rmse_mean']:.4f} ± {stats['masked_rmse_std']:.4f} (range: {stats['masked_rmse_min']:.4f} - {stats['masked_rmse_max']:.4f})")
        print(f"  Cropped LPIPS (All): {stats['cropped_lpips_mean']:.3f} ± {stats['cropped_lpips_std']:.3f} (range: {stats['cropped_lpips_min']:.3f} - {stats['cropped_lpips_max']:.3f})")
        print(f"  Cropped LPIPS (Reliable): {stats['reliable_cropped_lpips_mean']:.3f} ± {stats['reliable_cropped_lpips_std']:.3f} ({stats['reliable_cropped_lpips_count']}/{len(results[config_name])} poses = {stats['reliable_cropped_lpips_ratio']:.1%})")
        print(f"  Mask Coverage: {stats['mask_coverage_mean']:.1%} ± {stats['mask_coverage_std']:.1%} (range: {stats['mask_coverage_min']:.1%} - {stats['mask_coverage_max']:.1%})")
        print(f"  Valid Pixels: {stats['valid_pixels_mean']:.0f} ± {stats['valid_pixels_std']:.0f} (range: {stats['valid_pixels_min']} - {stats['valid_pixels_max']})")
        print(f"  Original Gaussians: {stats['wo_culled_count_mean']:.0f} ± {stats['wo_culled_count_std']:.0f} (range: {stats['wo_culled_count_min']} - {stats['wo_culled_count_max']})")
        print(f"  Culled Gaussians: {stats['culled_count_mean']:.0f} ± {stats['culled_count_std']:.0f} (range: {stats['culled_count_min']} - {stats['culled_count_max']})")
        print(f"  Culling Ratio: {stats['culling_ratio_mean']:.1f}% ± {stats['culling_ratio_std']:.1f}%")
        print(f"  Distance: {stats['distance_mean']:.2f} ± {stats['distance_std']:.2f}m (range: {stats['distance_min']:.2f} - {stats['distance_max']:.2f})")
        print()
    
    print(f"Results saved to:")
    print(f"  - Detailed: {detailed_file}")
    print(f"  - Summary: {summary_file}")
    if save_images and len(poses) > 0:  # Only print image info if poses were processed and images were saved
        print(f"  - Stacked comparison images: {images_output_dir}")
        print(f"  - Ground truth masks: {masks_output_dir}")
        for config in eval_configs:
            print(f"    - {config['name']} (GT|Culled|Difference|Heatmap): {culled_images_dirs[config['name']]}")
    elif not save_images:
        print(f"  - Images not saved (use --save-images to enable image output)")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate culling quality with configurable image saving")
    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=None,
        help="Pixel threshold for distance culling (default: None)"
    )
    parser.add_argument(
        "--save-images", 
        action="store_true", 
        default=True,
        help="Save rendered images (ground truth and culled) to disk (default: True)"
    )
    parser.add_argument(
        "--no-save-images", 
        dest="save_images", 
        action="store_false",
        help="Disable saving rendered images to disk"
    )

    parser.add_argument(
        "--lpips",
        action="store_true",
        default=False,
        help="Measure LPIPS (default: False)"
    )

    args = parser.parse_args()
    
    # Load config
    cfg = Config()
    # read the template of yaml from file
    template_path = "./configs/actorshq_stream.toml"
    config_from_file = load_config_from_toml(template_path)
    cfg = merge_config(cfg, config_from_file)
    if args.pixel_threshold is not None:
        cfg.pixel_threshold = args.pixel_threshold

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
    cfg.exp_name = exp_name
    
    cfg.disable_viewer = False
    iter = cfg.max_steps
    frame_id = 0
    
    print(f"Image saving: {'Enabled' if args.save_images else 'Disabled'}")
    
    # Run evaluation
    evaluate_culling_quality(frame_id, iter, cfg, exp_name, measure_lpips=args.lpips, save_images=args.save_images)

if __name__ == "__main__":
    main()


"""
Example usage:

python scripts/ablation/evaluate_quality_of_culling.py --pixel-threshold 0.01   # Set pixel threshold to 0.01, override the default value in the config file
python scripts/ablation/evaluate_quality_of_culling.py --save-images            # Explicitly save images  
python scripts/ablation/evaluate_quality_of_culling.py --no-save-images         # Skip saving images
python scripts/ablation/evaluate_quality_of_culling.py --lpips                  # To measure LPIPS, you need to pass --lpips flag. This takes a lot of time.

This will:
1. Load the config and checkpoint
2. Read poses from {result_dir}/viewer_poses/viewer_poses.json
3. For each pose, render images with different culling combinations:
   - Ground truth (no culling)
   - Frustum culling only
   - Distance culling only  
   - Both frustum + distance culling
4. Calculate PSNR, SSIM, LPIPS, masked PSNR, masked SSIM, and cropped LPIPS metrics using torchmetrics comparing culled images to ground truth
5. Use alpha channel from ground truth (no culling) render as mask for masked PSNR calculation
6. Optionally save ground truth masks to {result_dir}/viewer_poses/evaluation_images/gt_masks/ (if --save-images)
7. Optionally save stacked comparison images (GT|Culled|Difference|Heatmap) to {result_dir}/viewer_poses/evaluation_images/ organized by culling type (if --save-images)
8. Save detailed results (pose_id-keyed with psnr, ssim, lpips, masked_psnr, masked_ssim, cropped_lpips, culling_ratio, distance_to_bbox, num_valid_pixels) and summary statistics to {result_dir}/viewer_poses/culling_evaluation/
"""
