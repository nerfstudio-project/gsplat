#!/usr/bin/env python3
"""
Evaluate the quantitative quality (PSNR and SSIM) of Gaussian merging methods against ground truth.
This script reads saved viewer poses and renders images with different merging configurations.
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
import cv2

# Add parent directories to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gsplat.rendering import rasterization
from stream.culling import frustum_culling
from stream.merging import merge_gaussians

from examples.config import Config, load_config_from_toml, merge_config
from examples.utils import get_color, Metrics
from scripts.utils import set_result_dir, load_checkpoint, load_poses

import logging
log = logging.getLogger() # Use root logger
log.setLevel(logging.DEBUG)

# Create the handler for console output (stdout)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Log INFO and higher levels to console
console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s - %(funcName)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Create the handler for file output
file_handler = logging.FileHandler(f"{__name__}.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s - %(funcName)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add the handlers to the logger
log.addHandler(console_handler)
log.addHandler(file_handler)

def save_comparison_image_with_overlay(
    gt_image: torch.Tensor,
    processed_image: torch.Tensor,
    save_path: str,
    pose_id: int,
    distance_to_bbox: float,
    metrics: Dict
) -> None:
    """
    Save a comparison image with GT | Processed | Error Heatmap and overlay metrics.
    
    Args:
        gt_image: Ground truth image tensor [1, H, W, 3] in [0, 1] range
        processed_image: Processed image tensor [1, H, W, 3] in [0, 1] range  
        save_path: Path to save the comparison image
        pose_id: Pose identifier to overlay
        distance_to_bbox: Distance to bounding box to overlay
        metrics: Metrics dictionary to overlay
    """
    # Convert to numpy arrays
    gt_image_np = (gt_image[0].cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]
    processed_image_np = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]
    
    # Calculate difference map for visualization  
    gt_image_float = gt_image[0].cpu().numpy()  # [H, W, 3] in [0, 1]
    processed_image_float = processed_image[0].cpu().numpy()  # [H, W, 3] in [0, 1]
    
    # Absolute difference
    diff_abs = np.abs(gt_image_float - processed_image_float)  # [H, W, 3]
    
    # Create error heatmap (convert to grayscale then to heatmap)
    diff_gray = np.mean(diff_abs, axis=2)  # [H, W] average across RGB
    diff_heatmap = plt.cm.hot(np.clip(diff_gray * 10.0, 0.0, 1.0))[:, :, :3]  # Apply hot colormap
    diff_heatmap_np = (diff_heatmap * 255).astype(np.uint8)  # [H, W, 3]

    # Stack GT | Processed | Heatmap horizontally
    stacked_image = np.concatenate([gt_image_np, processed_image_np, diff_heatmap_np], axis=1)  # [H, 3*W, 3]
    
    # Convert RGB to BGR for OpenCV (OpenCV uses BGR by default)
    stacked_image_bgr = cv2.cvtColor(stacked_image, cv2.COLOR_RGB2BGR)
    
    # Font settings for OpenCV (increased size)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0  # Increased from 0.6
    font_thickness = 3  # Increased from 2
    text_color = (255, 255, 255)  # White text in BGR
    bg_color = (0, 0, 0)  # Black background in BGR
    
    # Overlay metrics text over the heatmap section (rightmost third)
    height, width = stacked_image.shape[:2]
    section_width = width // 3
    heatmap_start_x = 2 * section_width  # Start of heatmap section
    
    text_lines = [
        f"Pose ID: {pose_id}",
        f"Distance: {distance_to_bbox:.2f} m",
        f"mPSNR: {metrics['masked_psnr']:.2f} dB",
        f"PSNR: {metrics['psnr']:.2f} dB",
    ]
    
    # Position text over heatmap section
    x_offset = heatmap_start_x + 15  # 15 pixels from start of heatmap section
    y_offset = 100  # Increased from 30
    line_height = 100  # Increased from 35
    padding = 8  # Increased padding
    
    for i, text in enumerate(text_lines):
        y_pos = y_offset + i * line_height
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Draw semi-transparent background rectangle
        overlay = stacked_image_bgr.copy()
        cv2.rectangle(overlay, 
                     (x_offset - padding, y_pos - text_height - padding),
                     (x_offset + text_width + padding, y_pos + baseline + padding),
                     bg_color, -1)
        
        # Blend with original image for transparency effect
        alpha = 0.8  # Slightly more opaque for better readability
        cv2.addWeighted(overlay, alpha, stacked_image_bgr, 1 - alpha, 0, stacked_image_bgr)
        
        # Draw white text
        cv2.putText(stacked_image_bgr, text, (x_offset, y_pos), font, font_scale, text_color, font_thickness)
    
    # Save the image using OpenCV
    cv2.imwrite(save_path, stacked_image_bgr)


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
    apply_merging: bool = False,
    merging_config: Dict = None
) -> Tuple[torch.Tensor, torch.Tensor, int, int, float, Dict]:
    """Render an image for a given pose with optional frustum culling and merging.
    Args:
        means: [N, 3]
        quats: [N, 4]
        scales: [N, 3]
        opacities: [N]
        colors: [N, K, 3]
        pose_data: Dict
        device: torch.device
        cfg: Config
        apply_frustum_culling: bool
        apply_merging: bool
        merging_config: Dict - configuration for merging
    Returns:
        rendered_image: [1, H, W, 3]
        rendered_alpha: [1, H, W, 1]
        num_original: int - Number of Gaussians before processing
        num_after_processing: int - Number of Gaussians after processing
        processing_ratio: float - Percentage of change in Gaussian count
        merge_info: Dict - information about merging process
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
    
    # Apply merging if requested
    merge_info = {}
    if apply_merging and merging_config:
        try:
            merged_means, merged_quats, merged_scales, merged_opacities, merged_colors, merge_info = merge_gaussians(
                render_means, render_quats, render_scales, render_opacities, render_colors,
                viewmat, K_matrix, width, height,
                **merging_config
            )
            render_means = merged_means
            render_quats = merged_quats
            render_scales = merged_scales
            render_opacities = merged_opacities
            render_colors = merged_colors
        except RecursionError as e:
            log.error(f"    WARNING: RecursionError during merging - skipping merging for this pose")
            log.error(f"    Error: {str(e)[:100]}...")
            # Continue without merging - use original Gaussians
            merge_info = {"error": "RecursionError", "error_message": str(e)[:200]}
    
    # Handle case where all Gaussians are processed away
    if render_means.shape[0] == 0:
        log.warning(f"Warning: All Gaussians processed away, returning background image")
        background_color = torch.tensor(get_color("white"), dtype=torch.float32, device=device) / 255.0
        background_image = background_color.view(1, 1, 1, 3).expand(1, height, width, 3)
        background_alpha = torch.ones((1, height, width, 1), dtype=torch.float32, device=device)
        num_after_processing = 0
        processing_ratio = 100.0
        return background_image, background_alpha, num_original, num_after_processing, processing_ratio, merge_info
    
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
    
    num_after_processing = render_means.shape[0]
    # For merging, we show reduction in count (positive means reduction)
    processing_ratio = (num_original - num_after_processing) / num_original * 100.0
    
    return rendered_image, rendered_alpha, num_original, num_after_processing, processing_ratio, merge_info


def evaluate_merging_quality(frame_id, iter, cfg: Config, exp_name, sub_exp_name=None, save_images=True, measure_lpips=False, merge_accelerated=False):
    """Main evaluation function for merging quality."""

    cfg.data_dir = os.path.join(cfg.actorshq_data_dir, f"{frame_id}", f"resolution_{cfg.resolution}")
    cfg.exp_name = exp_name
    if sub_exp_name is not None:
        cfg.exp_name = f"{exp_name}_{sub_exp_name}"
    cfg.run_mode = "render"
    cfg.scene_id = frame_id
    set_result_dir(cfg, exp_name=exp_name, sub_exp_name=sub_exp_name)
    ckpt = os.path.join(f"{cfg.result_dir}/ckpts/ckpt_{iter - 1}_rank0.pt")
    cfg.ckpt = ckpt
    method = "cuda" if merge_accelerated else "torch"

    log.info(f"  Checkpoint: {cfg.ckpt}")
    log.info(f"  Result dir: {cfg.result_dir}")

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
    
    log.info(f"Using custom rendering resolution: {cfg.render_width}x{cfg.render_height}")
    log.info(f"Original pose resolution will be scaled accordingly")
    
    if len(poses) == 0:
        raise ValueError("No poses found for evaluation")
    
    # Define merging configurations
    merging_configs = {
        "knn_weighted_px": {
            "pixel_size_threshold": cfg.pixel_threshold,
            "use_pixel_area": False,
            "clustering_method": "knn",
            "accelerated": merge_accelerated,
            "merge_strategy": "weighted_mean",
            "clustering_kwargs": {"k_neighbors": 3, "max_distance": 0.1, "min_cluster_size": 2},
            "merge_kwargs": {"weight_by_opacity": True}
        },
        "knn_weighted_area": {
            "pixel_area_threshold": cfg.pixel_threshold,
            "use_pixel_area": True,
            "clustering_method": "knn",
            "accelerated": merge_accelerated,
            "merge_strategy": "weighted_mean",
            "clustering_kwargs": {"k_neighbors": 3, "max_distance": 0.1, "min_cluster_size": 2},
            "merge_kwargs": {"weight_by_opacity": True}
        },
        "knn_moment_px": {
            "pixel_size_threshold": cfg.pixel_threshold,
            "use_pixel_area": False,
            "clustering_method": "knn",
            "accelerated": merge_accelerated,
            "merge_strategy": "moment_matching",
            "clustering_kwargs": {"k_neighbors": 3, "max_distance": 0.1, "min_cluster_size": 2},
            "merge_kwargs": {"preserve_volume": True}
        },
        "knn_moment_area": {
            "pixel_area_threshold": cfg.pixel_threshold,
            "use_pixel_area": True,
            "clustering_method": "knn",
            "accelerated": merge_accelerated,
            "merge_strategy": "moment_matching",
            "clustering_kwargs": {"k_neighbors": 3, "max_distance": 0.1, "min_cluster_size": 2},
            "merge_kwargs": {"preserve_volume": True}
        },
        "dbscan_weighted_px": {
            "pixel_size_threshold": cfg.pixel_threshold,
            "use_pixel_area": False,
            "clustering_method": "dbscan",
            "accelerated": merge_accelerated,
            "merge_strategy": "weighted_mean",
            "clustering_kwargs": {"eps": 0.1, "min_samples": 2},
            "merge_kwargs": {"weight_by_opacity": True}
        },
        "dbscan_weighted_area": {
            "pixel_area_threshold": cfg.pixel_threshold,
            "use_pixel_area": True,
            "clustering_method": "dbscan",
            "accelerated": merge_accelerated,
            "merge_strategy": "weighted_mean",
            "clustering_kwargs": {"eps": 0.1, "min_samples": 2},
            "merge_kwargs": {"weight_by_opacity": True}
        },
        "center_in_pixel_weighted_px": {
            "pixel_size_threshold": cfg.pixel_threshold,
            "use_pixel_area": False,
            "clustering_method": "center_in_pixel",
            "accelerated": merge_accelerated,
            "merge_strategy": "weighted_mean",
            "clustering_kwargs": {"depth_threshold": 0.1, "min_cluster_size": 2},
            "merge_kwargs": {"weight_by_opacity": True}
        },
        "center_in_pixel_weighted_area": {
            "pixel_area_threshold": cfg.pixel_threshold,
            "use_pixel_area": True,
            "clustering_method": "center_in_pixel",
            "accelerated": merge_accelerated,
            "merge_strategy": "weighted_mean",
            "clustering_kwargs": {"depth_threshold": 0.1, "min_cluster_size": 2},
            "merge_kwargs": {"weight_by_opacity": True}
        }
    }

    eval_tests = ["center_in_pixel_weighted_area"]

    # Evaluation configurations - Focus on merging only (no frustum culling)
    # Include clustering parameters in names for better result tracking
    eval_configs = []
    for test in eval_tests:
        merge_config = merging_configs[test]
        if "center_in_pixel" in test:
            px_or_area = "px" if "px" in test else "area"
            merge_strategy = merging_configs[test]["merge_strategy"]
            eval_configs.append({
                "name": f"merge_{test}",
                "parameters" : merge_config,
                "param_str": f"depth{merge_config['clustering_kwargs']['depth_threshold']}_minc{merge_config['clustering_kwargs']['min_cluster_size']}_merge_{merge_strategy}_{px_or_area}_{cfg.pixel_threshold}",
                "frustum": False, "merging": True, "config": merging_configs[test]
            })
        elif "knn" in test:
            px_or_area = "px" if "px" in test else "area"
            merge_strategy = merging_configs[test]["merge_strategy"]
            eval_configs.append({
                "name": f"merge_{test}",
                "parameters": merge_config,
                "param_str": f"k{merging_configs[test]['clustering_kwargs']['k_neighbors']}_maxd{merging_configs[test]['clustering_kwargs']['max_distance']}_minc{merging_configs[test]['clustering_kwargs']['min_cluster_size']}_merge_{merge_strategy}_{px_or_area}_{cfg.pixel_threshold}",
                "frustum": False, "merging": True, "config": merging_configs[test]
            })
        elif "dbscan" in test:
            px_or_area = "px" if "px" in test else "area"
            merge_strategy = merging_configs[test]["merge_strategy"]
            eval_configs.append({
                "name": f"merge_{test}",
                "parameters": merge_config,
                "param_str": f"eps{merging_configs[test]['clustering_kwargs']['eps']}_mins{merging_configs[test]['clustering_kwargs']['min_samples']}_merge_{merge_strategy}_{px_or_area}_{cfg.pixel_threshold}",
                "frustum": False, "merging": True, "config": merging_configs[test]
            })
        else:
            raise ValueError(f"Unknown test: {test}")
        
    # Results storage - using pose_id as keys for easy lookup
    results = {config["name"]: {"parameters": config["parameters"], "param_str": config["param_str"], "results": {}} for config in eval_configs}
    
    # Create image output directories (only if saving images)
    images_output_dir = None
    merged_images_dirs = {}
    masks_output_dir = None
    
    if save_images:
        images_output_dir = os.path.join(poses_dir, f"render_hxw_{cfg.render_width}x{cfg.render_height}", f"static_merging/{method}/", "evaluation_images")
        os.makedirs(images_output_dir, exist_ok=True)
        
        # Create subdirectories for each configuration (stacked GT|Merged images)
        for config in eval_configs:
            if config["merging"]:  # Only create dirs for merging configs
                config_dir = os.path.join(images_output_dir, config["name"], config["param_str"])
                os.makedirs(config_dir, exist_ok=True)
                merged_images_dirs[config["name"]] = config_dir
        
        # Create directory for ground truth masks
        masks_output_dir = os.path.join(images_output_dir, "gt_masks")
        os.makedirs(masks_output_dir, exist_ok=True)
    
    log.info(f"\nStarting merging evaluation on {len(poses)} poses, pixel threshold: {cfg.pixel_threshold} px ...")
    start_time = time.time()
    
    for i, pose_data in enumerate(poses):
        log.info(f"\nEvaluating pose {i+1}/{len(poses)} (pose_id: {pose_data['pose_id']})")
        
        # Get pose_id and distance for consistent naming and result mapping
        pose_id = pose_data['pose_id']
        distance_to_bbox = pose_data['distance_to_bbox_center']
        
        # Render ground truth (no processing) - this will be the same for all configs
        gt_image, gt_alpha, original_count, _, _, _ = render_image(
            means, quats, scales, opacities, colors, pose_data, device, cfg,
            apply_frustum_culling=False, apply_merging=False
        ) # [1, H, W, 3], [1, H, W, 1]
        
        # Convert GT alpha to binary mask 
        # Use alpha > 0.5 threshold for binary mask
        gt_alpha = gt_alpha.squeeze(-1) # [1, H, W]
        gt_mask_binary = (gt_alpha > ALPHA_MASK_THRESHOLD).bool()  # [1, H, W]
        
        # Convert ground truth to numpy for image stacking (only if saving images)
        gt_image_np = (gt_image[0].cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]

        # Save GT masks for evaluation later
        if save_images and cfg.pixel_threshold == 2.0:  # Only save for baseline threshold
            gt_mask_binary_np = (gt_mask_binary[0, :, :].cpu().numpy() * 255.0).astype(np.uint8)  # [H, W]
            mask_path = os.path.join(masks_output_dir, f"pose_{pose_id:06d}_mask.png")
            imageio.imwrite(mask_path, gt_mask_binary_np)
        
        # Evaluate each configuration
        for config in eval_configs:
            render_start_time = time.time()
            processed_image, _, _, num_after_processing, processing_ratio, merge_info = render_image(
                means, quats, scales, opacities, colors, pose_data, device, cfg,
                apply_frustum_culling=config["frustum"],
                apply_merging=config["merging"],
                merging_config=config["config"]
            ) # [1, H, W, 3], [1, H, W, 1]
            render_time_ms = (time.time() - render_start_time) * 1000.0  # Convert to milliseconds
            log.info(f"    Time taken: {render_time_ms:.1f} ms")

            # Calculate all metrics using the Metrics class
            metric_results = metrics_calculator.calculate_all_metrics(
                processed_image, gt_image, gt_mask_binary, measure_lpips=measure_lpips
            )
            
            # Store results using pose_id as key
            results[config["name"]]["results"][pose_id] = {
                **metric_results,  # Include all metric results
                "distance_to_bbox": distance_to_bbox,
                "original_count": original_count,
                "processed_count": num_after_processing,
                "processing_ratio": processing_ratio,
                "render_time_ms": render_time_ms,
                "merge_info": merge_info if config["merging"] else {}
            }

            # Save comparison image with overlay (only if saving images and for merging configs)
            if save_images:
                assert config["merging"], f"Config {config['name']} is not merging"
                assert config["name"] in merged_images_dirs, f"Config {config['name']} not found in merged_images_dirs"
                
                # Save comparison image with metrics overlay
                stacked_image_path = os.path.join(merged_images_dirs[config["name"]], f"pose_{pose_id:06d}.png")
                save_comparison_image_with_overlay(
                    gt_image=gt_image,
                    processed_image=processed_image, 
                    save_path=stacked_image_path,
                    pose_id=pose_id,
                    distance_to_bbox=distance_to_bbox,
                    metrics=metric_results
                )
            
            # Print formatted metrics
            lpips_reliability = "✓" if metric_results['lpips_reliable'] else "⚠"
            processing_type = "Merged" if config["merging"] else "Culled"
            log.info(f"  {config['name'], config['param_str']}: PSNR={metric_results['psnr']:.2f}, SSIM={metric_results['ssim']:.4f}, LPIPS={metric_results['lpips']:.3f}, " 
            f"MaskedPSNR={metric_results['masked_psnr']:.2f}, MaskedSSIM={metric_results['masked_ssim']:.4f}, MaskedSE={metric_results['masked_se']:.2f}, MaskedRMSE={metric_results['masked_rmse']:.4f}, "
            f"CroppedLPIPS={metric_results['cropped_lpips']:.3f}{lpips_reliability}, "
            f"{processing_type}={processing_ratio:.1f}%, Coverage={metric_results['mask_coverage']:.1%}, Dist={distance_to_bbox:.3f}m, Valid={metric_results['num_valid_pixels']:.0f}, Time={render_time_ms:.1f} ms\n")
        
        # Progress update
        if (i + 1) % 10 == 0 or i == len(poses) - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(poses) - i - 1)
            log.info(f"Progress: {i+1}/{len(poses)} ({100*(i+1)/len(poses):.1f}%) - ETA: {eta:.1f}s")
    
    # Calculate and save summary statistics (simplified merge_info handling)
    for config_name, config_results in results.items():
        pose_results = config_results["results"]
        # Remove complex merge_info from pose_data to save space in JSON
        for pose_id in pose_results:
            if "merge_info" in pose_results[pose_id]:
                # Keep only basic merge statistics
                merge_info = pose_results[pose_id]["merge_info"]
                if merge_info:
                    simplified_merge_info = {
                        "original_count": merge_info.get("original_count", 0),
                        "final_count": merge_info.get("final_count", 0),
                        "reduction_ratio": merge_info.get("reduction_ratio", 0.0),
                        "iterations": merge_info.get("iterations", 0)
                    }
                    pose_results[pose_id]["merge_info"] = simplified_merge_info
    
    # Save detailed results
    output_dir = {}
    for config_name in results:
        output_dir[config_name] = os.path.join(poses_dir, f"render_hxw_{cfg.render_width}x{cfg.render_height}", f"static_merging/{method}/", "merging_evaluation", config_name, config_results["param_str"])
        os.makedirs(output_dir[config_name], exist_ok=True)
    
    # Save detailed results (pose_id-keyed format for easy lookup)
    for config_name in results:
        detailed_file = os.path.join(output_dir[config_name], f"detailed_results_{cfg.pixel_threshold}.json")
        with open(detailed_file, 'w') as f:
            json.dump(results[config_name], f, indent=2)
    
    # Print summary
    log.info(f"\n{'='*80}")
    log.info("MERGING QUALITY EVALUATION SUMMARY")
    log.info(f"{'='*80}")
    log.info(f"Total poses: {len(poses)}")
    log.info(f"Total time: {time.time() - start_time:.1f}s")
    log.info(f"\n")
    
    if save_images and len(poses) > 0:  # Only print image info if poses were processed and images were saved
        log.info(f"  - Ground truth masks: {masks_output_dir}")
        for config in eval_configs:
            if config["merging"]:
                assert config["name"] in merged_images_dirs, f"Config {config['name']} not found in merged_images_dirs"
                log.info(f"    - {config['name'], config['param_str']} (GT|Merged|Heatmap): {merged_images_dirs[config['name']]}")
    elif not save_images:
        log.info(f"  - Images not saved (use --save-images to enable image output)")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate merging quality with configurable image saving")
    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=None,
        help="Pixel threshold for merging (default: None)"
    )
    parser.add_argument(
        "--save-images", 
        action="store_true", 
        default=True,
        help="Save rendered images (ground truth and merged) to disk (default: True)"
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
    merge_accelerated = True
    
    log.info(f"Image saving: {'Enabled' if args.save_images else 'Disabled'}")
    log.info(f"Baseline pixel size threshold: {cfg.pixel_threshold}")    
    # Run evaluation
    evaluate_merging_quality(frame_id, iter, cfg, exp_name, measure_lpips=args.lpips, save_images=args.save_images, merge_accelerated=merge_accelerated)

if __name__ == "__main__":
    main()


"""
Example usage:

python scripts/merging/evaluate_static_merging.py --pixel-threshold 2.0     # Set pixel size threshold for merging
python scripts/merging/evaluate_static_merging.py --save-images             # Explicitly save images  
python scripts/merging/evaluate_static_merging.py --no-save-images          # Skip saving images
python scripts/merging/evaluate_static_merging.py --lpips                   # To measure LPIPS

This will:
1. Load the config and checkpoint
2. Read poses from {result_dir}/viewer_poses/viewer_poses.json
3. For each pose, render images with different merging configurations:
   - Ground truth (no processing)
   - KNN weighted merging (with k_neighbors, max_distance, min_cluster_size parameters in name)
   - DBSCAN weighted merging (with eps, min_samples parameters in name)
   - KNN moment matching merging (with k_neighbors, max_distance, min_cluster_size parameters in name)
4. Calculate PSNR, SSIM, LPIPS, masked PSNR, masked SSIM, and cropped LPIPS metrics comparing merged images to ground truth
5. Use alpha channel from ground truth (no processing) render as mask for masked PSNR calculation
6. Optionally save ground truth masks and stacked comparison images (GT|Merged|Heatmap)
7. Save detailed results and summary statistics with merging-specific metrics
"""
