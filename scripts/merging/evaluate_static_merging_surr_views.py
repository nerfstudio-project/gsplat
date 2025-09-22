#!/usr/bin/env python3
"""
Evaluate the viewpoint robustness of Gaussian merging methods.
This script takes a base pose, performs merging based on that pose, then evaluates 
the merged result on 4 surrounding poses to test view-consistency.
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

# Import render_image function and constants from static merging script to avoid duplication
from scripts.merging.evaluate_static_merging import render_image, ALPHA_MASK_THRESHOLD


def save_comparison_image_with_overlay(
    gt_image: torch.Tensor,
    processed_image: torch.Tensor,
    save_path: str,
    pose_id: int,
    distance_to_bbox: float,
    masked_psnr: float,
    pose_type: str = ""
) -> None:
    """
    Save a comparison image with GT | Processed | Error Heatmap and overlay metrics.
    
    Args:
        gt_image: Ground truth image tensor [1, H, W, 3] in [0, 1] range
        processed_image: Processed image tensor [1, H, W, 3] in [0, 1] range  
        save_path: Path to save the comparison image
        pose_id: Pose identifier to overlay
        distance_to_bbox: Distance to bounding box to overlay
        masked_psnr: Masked PSNR value to overlay
        pose_type: Type of pose (e.g., "base", "p0", "p1", etc.)
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
    font_scale = 2.0  # Large text
    font_thickness = 3  # Thick text
    text_color = (255, 255, 255)  # White text in BGR
    bg_color = (0, 0, 0)  # Black background in BGR
    
    # Overlay metrics text over the heatmap section (rightmost third)
    height, width = stacked_image.shape[:2]
    section_width = width // 3
    heatmap_start_x = 2 * section_width  # Start of heatmap section
    
    text_lines = [
        f"Pose: {pose_type}_{pose_id}" if pose_type else f"Pose ID: {pose_id}",
        f"Distance: {distance_to_bbox:.2f} m",
        f"mPSNR: {masked_psnr:.2f} dB"
    ]
    
    # Position text over heatmap section
    x_offset = heatmap_start_x + 15  # 15 pixels from start of heatmap section
    y_offset = 100  # Start position
    line_height = 100  # Space between lines
    padding = 8  # Padding around text
    
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


def generate_surrounding_poses(base_pose_data: Dict, displacement: float = 0.1, inward_angle: float = 0.05) -> List[Dict]:
    """
    Generate 4 poses around the base pose: right, left, top, bottom with slight inward angles.
    
    Args:
        base_pose_data: Original pose data dictionary
        displacement: Distance to move the camera in each direction (in world units)
        inward_angle: Small angle in radians to rotate camera inward toward subject (default: ~3 degrees)
        
    Returns:
        List of 4 new pose data dictionaries
    """
    base_c2w = np.array(base_pose_data["c2w_matrix"])
    
    # Extract camera position and orientation
    base_position = base_c2w[:3, 3]  # Camera position in world space
    base_rotation = base_c2w[:3, :3]  # Camera rotation matrix
    
    # Define displacement vectors in world coordinate system
    # Right/Left along world X-axis, Top/Bottom along world Y-axis
    world_displacements = [
        np.array([displacement, 0, 0]),   # Right (positive X in world space)
        np.array([-displacement, 0, 0]),  # Left (negative X in world space)
        np.array([0, displacement, 0]),   # Top (positive Y in world space)
        np.array([0, -displacement, 0]),  # Bottom (negative Y in world space)
    ]
    
    surrounding_poses = []
    pose_labels = ["right", "left", "top", "bottom"]
    
    # Rotation axes for inward angling (in world coordinates)
    rotation_axes = [
        np.array([0, 1, 0]),   # Right: rotate around Y-axis (yaw left)
        np.array([0, 1, 0]),   # Left: rotate around Y-axis (yaw right)  
        np.array([1, 0, 0]),   # Top: rotate around X-axis (pitch down)
        np.array([1, 0, 0]),   # Bottom: rotate around X-axis (pitch up)
    ]
    
    # Rotation directions for inward angling
    rotation_directions = [-1, 1, -1, 1]  # Negative angles for right/top, positive for left/bottom
    
    for i, (world_disp, rot_axis, rot_dir) in enumerate(zip(world_displacements, rotation_axes, rotation_directions)):
        # Create new camera position
        new_position = base_position + world_disp
        
        # Create rotation matrix for inward angling
        angle = rot_dir * inward_angle
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        if np.allclose(rot_axis, [0, 1, 0]):  # Y-axis rotation (yaw)
            inward_rotation = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
        else:  # X-axis rotation (pitch)
            inward_rotation = np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ])
        
        # Apply inward rotation to the base camera orientation
        new_rotation = inward_rotation @ base_rotation
        
        # Create new c2w matrix with new position and orientation
        new_c2w = np.eye(4)
        new_c2w[:3, :3] = new_rotation
        new_c2w[:3, 3] = new_position
        
        # Create new pose data
        new_pose_data = base_pose_data.copy()
        new_pose_data["c2w_matrix"] = new_c2w.tolist()
        new_pose_data["pose_id"] = f"{base_pose_data['pose_id']}_p{i}"
        new_pose_data["pose_label"] = f"p{i}_{pose_labels[i]}"
        
        # Update distance to bbox center (approximate)
        new_pose_data["distance_to_bbox_center"] = float(np.linalg.norm(new_position))
        
        surrounding_poses.append(new_pose_data)
    
    return surrounding_poses


def evaluate_viewpoint_robustness(frame_id, iter, cfg: Config, exp_name, sub_exp_name=None, save_images=True, measure_lpips=False, displacement=0.1, inward_angle=0.05):
    """Main evaluation function for viewpoint robustness of merging."""

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
    
    # Define merging configuration (center_in_pixel for robustness testing)
    merging_config = {
        "pixel_area_threshold": cfg.pixel_threshold,
        "use_pixel_area": True,
        "clustering_method": "center_in_pixel",
        "merge_strategy": "weighted_mean",
        "clustering_kwargs": {"depth_threshold": 0.1, "min_cluster_size": 2},
        "merge_kwargs": {"weight_by_opacity": True}
    }
    
    config_name = "center_in_pixel_weighted_area"
    param_str = f"depth{merging_config['clustering_kwargs']['depth_threshold']}_minc{merging_config['clustering_kwargs']['min_cluster_size']}_merge_weighted_mean_area_{cfg.pixel_threshold}"
    
    # Results storage - separate for base pose and surrounding poses
    results = {
        "base_pose": {"parameters": merging_config, "param_str": param_str, "results": {}},
        "surrounding_poses": {"parameters": merging_config, "param_str": param_str, "results": {}}
    }
    
    # Create image output directories (only if saving images)
    images_output_dir = None
    base_images_dir = None
    surrounding_images_dir = None
    masks_output_dir = None
    
    if save_images:
        images_output_dir = os.path.join(poses_dir, f"render_hxw_{cfg.render_width}x{cfg.render_height}", "static_merging_viewpoint_evaluation", "evaluation_images")
        os.makedirs(images_output_dir, exist_ok=True)
        
        # Create subdirectories for base pose and surrounding poses
        base_images_dir = os.path.join(images_output_dir, f"base_pose_{config_name}", param_str)
        surrounding_images_dir = os.path.join(images_output_dir, f"surr_poses_{config_name}", param_str)
        os.makedirs(base_images_dir, exist_ok=True)
        os.makedirs(surrounding_images_dir, exist_ok=True)
        
        # Create directory for ground truth masks
        masks_output_dir = os.path.join(images_output_dir, "gt_masks")
        os.makedirs(masks_output_dir, exist_ok=True)
    
    print(f"\nStarting viewpoint robustness evaluation on {len(poses)} poses, pixel threshold: {cfg.pixel_threshold} px ...")
    print(f"Configuration: {config_name} with {param_str}")
    start_time = time.time()
    
    for i, base_pose_data in enumerate(poses):
        print(f"\nEvaluating base pose {i+1}/{len(poses)} (pose_id: {base_pose_data['pose_id']})")
        
        base_pose_id = base_pose_data['pose_id']
        base_distance = base_pose_data['distance_to_bbox_center']
        
        # Step 1: Render ground truth for base pose and perform merging
        print(f"  Step 1: Merging Gaussians based on base pose {base_pose_id}")
        
        # Render GT for base pose
        gt_image_base, gt_alpha_base, original_count, _, _, _ = render_image(
            means, quats, scales, opacities, colors, base_pose_data, device, cfg,
            apply_frustum_culling=False, apply_merging=False
        )
        
        # Convert GT alpha to binary mask 
        gt_alpha_base = gt_alpha_base.squeeze(-1) # [1, H, W]
        gt_mask_binary_base = (gt_alpha_base > ALPHA_MASK_THRESHOLD).bool()  # [1, H, W]
        
        # Render with merging for base pose
        render_start_time = time.time()
        merged_image_base, _, _, num_after_processing, processing_ratio, merge_info = render_image(
            means, quats, scales, opacities, colors, base_pose_data, device, cfg,
            apply_frustum_culling=False, apply_merging=True, merging_config=merging_config
        )
        render_time_ms = (time.time() - render_start_time) * 1000.0
        
        # Skip this pose if merging failed
        if merge_info.get("error") == "RecursionError":
            print(f"  Skipping pose {base_pose_id} due to RecursionError in merging")
            continue
        
        # Calculate metrics for base pose
        metric_results_base = metrics_calculator.calculate_all_metrics(
            merged_image_base, gt_image_base, gt_mask_binary_base, measure_lpips=measure_lpips
        )
        
        # Store results for base pose
        results["base_pose"]["results"][base_pose_id] = {
            **metric_results_base,
            "distance_to_bbox": base_distance,
            "original_count": original_count,
            "processed_count": num_after_processing,
            "processing_ratio": processing_ratio,
            "render_time_ms": render_time_ms,
            "merge_info": merge_info
        }
        
        print(f"    Base pose metrics: PSNR={metric_results_base['psnr']:.2f}, mPSNR={metric_results_base['masked_psnr']:.2f}, Merged={processing_ratio:.1f}%")
        
        # Save comparison image for base pose
        if save_images:
            base_image_path = os.path.join(base_images_dir, f"pose_{base_pose_id:06d}.png")
            save_comparison_image_with_overlay(
                gt_image=gt_image_base,
                processed_image=merged_image_base, 
                save_path=base_image_path,
                pose_id=base_pose_id,
                distance_to_bbox=base_distance,
                masked_psnr=metric_results_base['masked_psnr'],
                pose_type="base"
            )
            
            # Save GT masks for base pose
            if cfg.pixel_threshold == 2.0:  # Only save for baseline threshold
                gt_mask_binary_np = (gt_mask_binary_base[0, :, :].cpu().numpy() * 255.0).astype(np.uint8)
                mask_path = os.path.join(masks_output_dir, f"pose_{base_pose_id:06d}_base_mask.png")
                imageio.imwrite(mask_path, gt_mask_binary_np)
        
        # Step 2: Get merged Gaussians from base pose for use in surrounding poses
        print(f"  Step 2: Using merged Gaussians to evaluate 4 surrounding poses")
        
        # Get the merged Gaussians by repeating the merging process (since we don't store them)
        # This is not ideal but necessary since render_image doesn't return the merged Gaussians
        c2w_matrix = torch.tensor(base_pose_data["c2w_matrix"], device=device, dtype=torch.float32)
        K_matrix_original = torch.tensor(base_pose_data["K_matrix"], device=device, dtype=torch.float32)
        
        # Adjust intrinsics for rendering resolution
        scale_x = cfg.render_width / base_pose_data["width"]
        scale_y = cfg.render_height / base_pose_data["height"]
        K_matrix = K_matrix_original.clone()
        K_matrix[0, 0] *= scale_x
        K_matrix[1, 1] *= scale_y
        K_matrix[0, 2] *= scale_x
        K_matrix[1, 2] *= scale_y
        
        viewmat = c2w_matrix.inverse()
        
        # Perform merging based on base pose to get merged Gaussians
        try:
            merged_means, merged_quats, merged_scales, merged_opacities, merged_colors, _ = merge_gaussians(
                means, quats, scales, opacities, colors,
                viewmat, K_matrix, cfg.render_width, cfg.render_height,
                **merging_config
            )
        except RecursionError:
            print(f"  RecursionError when extracting merged Gaussians, skipping surrounding poses")
            continue
        
        # Generate 4 surrounding poses (right, left, top, bottom with inward angles)
        surrounding_poses = generate_surrounding_poses(base_pose_data, displacement=displacement, inward_angle=inward_angle)
        
        # Step 3: Evaluate merged model on surrounding poses
        for j, surr_pose_data in enumerate(surrounding_poses):
            surr_pose_id = surr_pose_data['pose_id']
            surr_distance = surr_pose_data['distance_to_bbox_center']
            pose_label = surr_pose_data['pose_label']
            
            print(f"    Evaluating surrounding pose {j+1}/4: {pose_label} (id: {surr_pose_id})")
            
            # Render ground truth for surrounding pose (no merging)
            gt_image_surr, gt_alpha_surr, _, _, _, _ = render_image(
                means, quats, scales, opacities, colors, surr_pose_data, device, cfg,
                apply_frustum_culling=False, apply_merging=False
            )
            
            # Convert GT alpha to binary mask
            gt_alpha_surr = gt_alpha_surr.squeeze(-1)
            gt_mask_binary_surr = (gt_alpha_surr > ALPHA_MASK_THRESHOLD).bool()
            
            # Render using merged Gaussians from base pose (no additional merging)
            render_start_time = time.time()
            merged_image_surr, _, _, _, _, _ = render_image(
                merged_means, merged_quats, merged_scales, merged_opacities, merged_colors, 
                surr_pose_data, device, cfg,
                apply_frustum_culling=False, apply_merging=False
            )
            render_time_ms = (time.time() - render_start_time) * 1000.0
            
            # Calculate metrics for surrounding pose
            metric_results_surr = metrics_calculator.calculate_all_metrics(
                merged_image_surr, gt_image_surr, gt_mask_binary_surr, measure_lpips=measure_lpips
            )
            
            # Store results for surrounding pose
            results["surrounding_poses"]["results"][surr_pose_id] = {
                **metric_results_surr,
                "distance_to_bbox": surr_distance,
                "original_count": merged_means.shape[0],  # Using merged count as "original" for surrounding poses
                "processed_count": merged_means.shape[0],  # No additional processing
                "processing_ratio": 0.0,  # No additional processing
                "render_time_ms": render_time_ms,
                "base_pose_id": base_pose_id,
                "pose_label": pose_label,
                "merge_info": {"used_merged_from_base": True}
            }
            
            print(f"      {pose_label} metrics: PSNR={metric_results_surr['psnr']:.2f}, mPSNR={metric_results_surr['masked_psnr']:.2f}")
            
            # Save comparison image for surrounding pose
            if save_images:
                surr_image_path = os.path.join(surrounding_images_dir, f"pose_{surr_pose_id}.png")
                save_comparison_image_with_overlay(
                    gt_image=gt_image_surr,
                    processed_image=merged_image_surr,
                    save_path=surr_image_path,
                    pose_id=surr_pose_id,
                    distance_to_bbox=surr_distance,
                    masked_psnr=metric_results_surr['masked_psnr'],
                    pose_type=pose_label
                )
                
                # Save GT masks for surrounding poses
                if cfg.pixel_threshold == 2.0:  # Only save for baseline threshold
                    gt_mask_binary_np = (gt_mask_binary_surr[0, :, :].cpu().numpy() * 255.0).astype(np.uint8)
                    mask_path = os.path.join(masks_output_dir, f"pose_{surr_pose_id}_mask.png")
                    imageio.imwrite(mask_path, gt_mask_binary_np)
        
        # Progress update
        if (i + 1) % 5 == 0 or i == len(poses) - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(poses) - i - 1)
            print(f"Progress: {i+1}/{len(poses)} ({100*(i+1)/len(poses):.1f}%) - ETA: {eta:.1f}s")
    
    # Save detailed results
    output_dir = os.path.join(poses_dir, f"render_hxw_{cfg.render_width}x{cfg.render_height}", "static_merging_viewpoint_evaluation", "evaluation_results", config_name, param_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results for base poses and surrounding poses separately
    for result_type in ["base_pose", "surrounding_poses"]:
        detailed_file = os.path.join(output_dir, f"{result_type}_results_{cfg.pixel_threshold}.json")
        with open(detailed_file, 'w') as f:
            json.dump(results[result_type], f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("VIEWPOINT ROBUSTNESS EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total base poses evaluated: {len(results['base_pose']['results'])}")
    print(f"Total surrounding poses evaluated: {len(results['surrounding_poses']['results'])}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print()
    
    if save_images and len(results['base_pose']['results']) > 0:
        print(f"  - Base pose images: {base_images_dir}")
        print(f"  - Surrounding pose images: {surrounding_images_dir}")
        print(f"  - Ground truth masks: {masks_output_dir}")
    elif not save_images:
        print(f"  - Images not saved (use --save-images to enable image output)")
    
    print(f"\nDetailed results saved to: {output_dir}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate viewpoint robustness of Gaussian merging")
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
    parser.add_argument(
        "--displacement",
        type=float,
        default=0.1,
        help="Displacement distance for surrounding poses (default: 0.1)"
    )
    parser.add_argument(
        "--inward-angle",
        type=float,
        default=0.05,
        help="Inward angle in radians for surrounding poses to point toward subject (default: 0.05, ~3 degrees)"
    )

    args = parser.parse_args()
    
    # Load config
    cfg = Config()
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
    print(f"Pixel threshold: {cfg.pixel_threshold}")
    print(f"Surrounding pose displacement: {args.displacement}")
    print(f"Inward angle: {args.inward_angle} rad (~{args.inward_angle * 180 / 3.14159:.1f} degrees)")
    
    # Run evaluation
    evaluate_viewpoint_robustness(frame_id, iter, cfg, exp_name, measure_lpips=args.lpips, save_images=args.save_images, 
                                displacement=args.displacement, inward_angle=args.inward_angle)

if __name__ == "__main__":
    main()


"""
Example usage:

python scripts/merging/evaluate_viewpoint_robustness_merging.py --pixel-threshold 2.0
python scripts/merging/evaluate_viewpoint_robustness_merging.py --save-images --lpips
python scripts/merging/evaluate_viewpoint_robustness_merging.py --displacement 0.2 --inward-angle 0.05

This script will:
1. For each base pose, perform merging and evaluate quality vs ground truth
2. Generate 4 surrounding poses (right, left, top, bottom) with inward angles around each base pose
3. Use the merged Gaussians from step 1 to render images for the 4 surrounding poses
4. Compare rendered images from merged Gaussians with ground truth for each surrounding poses  
5. Save all comparison images and detailed metrics
6. Generate summary statistics for viewpoint robustness analysis

Results are saved in viewpoint_robustness_merging/ directory with separate folders for:
- base_pose/: Results from merging evaluation on original poses
- surrounding_poses/: Results from evaluating merged model on surrounding poses
- evaluation_images/: All comparison images with metrics overlays
- evaluation_results/: JSON files with detailed metrics
"""
