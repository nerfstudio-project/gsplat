#!/usr/bin/env python3
"""
Simple pixel difference analysis script.
Reads one pose at a time from evaluation images and creates distribution plots of absolute pixel differences.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio
import re

WORK_DIR = "/main/rajrup/Dropbox/Project/GsplatStream/gsplat"

def load_stacked_image(image_path):
    """
    Load a stacked image (GT|Culled|Heatmap) and extract GT and culled portions.
    
    Returns:
        gt_image: Ground truth image [H, W, 3]
        culled_image: Culled image [H, W, 3]
    """
    stacked_image = imageio.imread(image_path)  # [H, 3*W, 3]
    
    height, total_width, channels = stacked_image.shape
    width = total_width // 3  # Each section is 1/3 of total width
    
    # Extract GT and culled sections
    gt_image = stacked_image[:, 0:width, :]                    # GT
    culled_image = stacked_image[:, width:2*width, :]          # Culled
    
    return gt_image, culled_image

def load_gt_mask(mask_path):
    """
    Load ground truth mask for the pose.
    
    Returns:
        mask: [H, W] - boolean mask where True indicates foreground pixels
    """
    if not os.path.exists(mask_path):
        print(f"Warning: GT mask not found at {mask_path}")
        return None
    
    mask_image = imageio.imread(mask_path)  # [H, W] grayscale
    # Convert to boolean mask (assuming mask is binary with 255 for foreground)
    mask = mask_image > 127  # [H, W] boolean
    
    return mask

def compute_absolute_pixel_differences(gt_image, culled_image, gt_mask):
    """
    Compute absolute pixel differences between GT and culled images.
    Only considers pixels within the ground truth mask if provided.
    
    Returns:
        abs_diff_per_pixel: [H, W] - mean absolute difference per pixel across RGB in 0-255 range
        valid_mask: [H, W] - boolean mask indicating which pixels were considered
    """
    # Keep in 0-255 range as float32
    gt_float = gt_image.astype(np.float32)
    culled_float = culled_image.astype(np.float32)
    
    # Absolute difference per channel
    abs_diff = np.abs(gt_float - culled_float)  # [H, W, 3]
    
    # Mean absolute difference per pixel (average across RGB)
    abs_diff_per_pixel = np.mean(abs_diff, axis=2)  # [H, W]
    
    # Apply ground truth mask if provided
    if gt_mask is not None:
        valid_mask = gt_mask
    else:
        # raise error
        raise ValueError("GT mask is required for pixel difference analysis")
    
    return abs_diff_per_pixel, valid_mask

def create_distribution_plot(abs_diff_per_pixel, valid_mask, pose_id, pixel_threshold, save_path):
    """
    Create a distribution plot of absolute pixel differences for a single pose.
    Only considers pixels within the valid mask (ground truth mask).
    """
    # Apply mask to get only valid pixels
    masked_diff_values = abs_diff_per_pixel[valid_mask]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(masked_diff_values) > 0:
        # Histogram with 255 bins and range 0-255
        ax.hist(masked_diff_values, bins=255, range=(0, 255), alpha=0.7, density=True, color='blue', edgecolor='black', linewidth=0.1)
        
        # Add statistics text
        mean_diff = np.mean(masked_diff_values)
        std_diff = np.std(masked_diff_values)
        max_diff = np.max(masked_diff_values)
        median_diff = np.median(masked_diff_values)
        
        # Calculate mask coverage
        total_pixels = abs_diff_per_pixel.size
        valid_pixels = np.sum(valid_mask)
        mask_coverage = (valid_pixels / total_pixels) * 100
        
        stats_text = f'Mean: {mean_diff:.2f}\nStd: {std_diff:.2f}\nMax: {max_diff:.2f}\nMedian: {median_diff:.2f}\nMask Coverage: {mask_coverage:.1f}%\nValid Pixels: {valid_pixels}'
        ax.text(0.7, 0.8, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Labels and title
        ax.set_xlabel('Absolute Pixel Difference [0-255]', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Pixel Difference Distribution (Masked)\nPose {pose_id:06d} - Pixel Threshold {pixel_threshold}', fontsize=14)
        ax.set_xlim(0, 255)
    else:
        # Handle case where no valid pixels in mask
        ax.text(0.5, 0.5, 'No valid pixels in mask\n(no differences to plot)', 
                transform=ax.transAxes, fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax.set_xlabel('Absolute Pixel Difference [0-255]', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Pixel Difference Distribution (Masked)\nPose {pose_id:06d} - Pixel Threshold {pixel_threshold}', fontsize=14)
        ax.set_xlim(0, 255)
    
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distribution plot to: {save_path}")

def process_single_pose(image_path, pixel_threshold, save_dir, gt_masks_dir):
    """
    Process a single pose image and create its distribution plot.
    """
    # Extract pose ID from filename
    pose_id = int(re.search(r'pose_(\d+)\.png', os.path.basename(image_path)).group(1))
    print(f"Processing pose {pose_id}")
    
    try:
        # Load and extract GT and culled images
        gt_image, culled_image = load_stacked_image(image_path)
        # print(f"GT image shape: {gt_image.shape}")
        # print(f"Culled image shape: {culled_image.shape}")
        
        # Load ground truth mask
        mask_path = os.path.join(gt_masks_dir, f"pose_{pose_id:06d}_mask.png")
        gt_mask = load_gt_mask(mask_path)
        
        if gt_mask is None:
            print(f"Skipping pose {pose_id} - no ground truth mask found")
            return
        
        # print(f"GT mask shape: {gt_mask.shape}")
        print(f"GT mask coverage: {np.sum(gt_mask) / gt_mask.size * 100:.1f}%")
        
        # Compute absolute pixel differences with mask
        abs_diff_per_pixel, valid_mask = compute_absolute_pixel_differences(gt_image, culled_image, gt_mask)
        
        # Create save path
        save_path = os.path.join(save_dir, f"pose_{pose_id:06d}_pixel_diff_distribution.png")
        
        # Create and save distribution plot
        create_distribution_plot(abs_diff_per_pixel, valid_mask, pose_id, pixel_threshold, save_path)
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def discover_pixel_thresholds(evaluation_images_dir):
    """
    Discover available pixel thresholds from the distance_only_px_* directories.
    """
    pattern = os.path.join(evaluation_images_dir, "distance_only_px_*")
    dirs = glob.glob(pattern)
    
    pixel_thresholds = []
    for dir_path in dirs:
        dir_name = os.path.basename(dir_path)
        match = re.search(r'distance_only_px_([0-9]+\.?[0-9]*)', dir_name)
        if match:
            pixel_thresholds.append(float(match.group(1)))
    
    return sorted(pixel_thresholds)


def main():
    # Configuration
    model_name = "actorshq_l1_0.5_ssim_0.5_alpha_1.0"
    actor_name = "Actor01"
    seq_name = "Sequence1"
    resolution = 4
    frame_id = 0
    # render_height = 1422
    # render_width = 2048
    render_height = 356 # 1/4 of 1422
    render_width = 512  # 1/4 of 2048
    
    print(f"{'='*80}")
    print(f"Processing pixel differences for {model_name}/{actor_name}/{seq_name}/{resolution}x/{frame_id}")
    print(f"Render resolution: {render_width}x{render_height}")
    print(f"{'='*80}")
    
    # Paths
    evaluation_images_dir = os.path.join(
        WORK_DIR, 
        f"results/{model_name}/{actor_name}/{seq_name}/resolution_{resolution}/{frame_id}/viewer_poses/render_hxw_{render_width}x{render_height}/evaluation_images"
    )
    
    gt_masks_dir = os.path.join(evaluation_images_dir, "gt_masks")
    
    save_dir = os.path.join(
        WORK_DIR, 
        f"scripts/plots/pixel_difference_analysis/{model_name}/{actor_name}_{seq_name}_{resolution}x_{frame_id}/render_hxw_{render_width}x{render_height}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Reading evaluation images from: {evaluation_images_dir}")
    print(f"Reading GT masks from: {gt_masks_dir}")
    print(f"Saving distribution plots to: {save_dir}")
    
    # Check if GT masks directory exists
    if not os.path.exists(gt_masks_dir):
        print(f"Error: GT masks directory not found: {gt_masks_dir}")
        print("Please run evaluate_quality_of_culling.py first to generate GT masks")
        return
    
    # Discover available pixel thresholds
    pixel_thresholds = discover_pixel_thresholds(evaluation_images_dir)
    # sort in descending order
    pixel_thresholds.sort(reverse=True)
    
    if not pixel_thresholds:
        print("No pixel threshold directories found!")
        return
    
    print(f"Found pixel thresholds: {pixel_thresholds}")
    
    # Process each threshold
    for threshold in pixel_thresholds:
        if threshold == 0.0:
            continue

        print(f"\n{'='*60}")
        print(f"Processing pixel threshold: {threshold}")
        print(f"{'='*60}")
        
        # Create threshold-specific directory
        threshold_save_dir = os.path.join(save_dir, f"pixel_threshold_{threshold}")
        os.makedirs(threshold_save_dir, exist_ok=True)
        
        # Find all pose images for this threshold
        threshold_dir = os.path.join(evaluation_images_dir, f"distance_only_px_{threshold}")
        if not os.path.exists(threshold_dir):
            print(f"Directory not found: {threshold_dir}")
            continue
        
        image_pattern = os.path.join(threshold_dir, "pose_*.png")
        image_files = sorted(glob.glob(image_pattern))
        
        if not image_files:
            print(f"No pose images found in {threshold_dir}")
            continue
        
        print(f"Found {len(image_files)} pose images")
        
        # Process each pose image
        for image_file in image_files:
            process_single_pose(image_file, threshold, threshold_save_dir, gt_masks_dir)
    
    print(f"\n{'='*80}")
    print("Pixel difference analysis completed!")
    print(f"All distribution plots saved under: {save_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
