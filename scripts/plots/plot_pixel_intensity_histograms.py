#!/usr/bin/env python3
"""
Pixel intensity histogram analysis script.
Reads evaluation images from distance culling and creates side-by-side histograms comparing
ground truth (no culling) vs rendering after distance-based culling for masked regions.

Supports multiple intensity extraction methods:
- "luminance": Proper luminance conversion (0.299*R + 0.587*G + 0.114*B) [RECOMMENDED]
  Best for analyzing perceived brightness distribution
- "mean": Simple RGB mean (R+G+B)/3
  Good approximation for brightness analysis  
- "flatten": All RGB values flattened together
  Treats each color channel value independently (less meaningful for intensity analysis)
- "separate": Separate R, G, B histograms 
  Most comprehensive for color-specific analysis and detecting channel artifacts

From an image processing perspective:
- Use "luminance" for standard intensity/brightness analysis
- Use "separate" for comprehensive color analysis  
- Use "mean" for simpler brightness approximation
- Avoid "flatten" unless specifically analyzing raw color channel distributions
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio
import re

WORK_DIR = "/ssd1/rajrup/Project/gsplat"

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

def extract_masked_pixel_intensities(image, mask, method="luminance"):
    """
    Extract pixel intensities from masked regions of the image.
    
    Args:
        image: [H, W, 3] - RGB image in 0-255 range
        mask: [H, W] - boolean mask where True indicates valid pixels
        method: str - "luminance", "mean", "flatten", or "separate"
    
    Returns:
        For "luminance"/"mean": [N] - single intensity per pixel
        For "flatten": [N*3] - flattened RGB values 
        For "separate": dict with "R", "G", "B" keys, each [N]
    """
    if mask is None:
        raise ValueError("GT mask is required for pixel intensity analysis")
    
    # Apply mask to get valid pixels
    masked_pixels = image[mask]  # [N_valid_pixels, 3]
    
    if method == "luminance":
        # Proper luminance conversion (perceptually accurate)
        intensities = 0.299 * masked_pixels[:, 0] + 0.587 * masked_pixels[:, 1] + 0.114 * masked_pixels[:, 2]
    elif method == "mean":
        # Simple mean across RGB channels
        intensities = np.mean(masked_pixels, axis=1)  # [N_valid_pixels]
    elif method == "flatten":
        # Flatten all RGB values (original approach)
        intensities = masked_pixels.flatten()  # [N_valid_pixels * 3]
    elif method == "separate":
        # Return separate R, G, B arrays
        intensities = {
            "R": masked_pixels[:, 0],
            "G": masked_pixels[:, 1], 
            "B": masked_pixels[:, 2]
        }
    else:
        raise ValueError(f"Invalid method: {method}. Choose from 'luminance', 'mean', 'flatten', 'separate'")
    
    return intensities

def create_side_by_side_histogram(gt_intensities, culled_intensities, pose_id, pixel_threshold, save_path, method="luminance"):
    """
    Create side-by-side histograms comparing GT and culled pixel intensity distributions.
    
    Args:
        gt_intensities: [N] or dict - GT pixel intensities in 0-255 range
        culled_intensities: [M] or dict - Culled pixel intensities in 0-255 range
        pose_id: int - pose identifier
        pixel_threshold: float - pixel threshold used for culling
        save_path: str - path to save the plot
        method: str - intensity extraction method used
    """
    if method == "separate":
        # Create 2x3 subplots for RGB channels
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
        channels = ['R', 'G', 'B']
        colors = ['red', 'green', 'blue']
        
        for i, (channel, color) in enumerate(zip(channels, colors)):
            # GT histogram for this channel
            ax_gt = axes[0, i]
            ax_culled = axes[1, i]
            
            # Common histogram parameters
            bins = np.arange(0, 257, 1)
            alpha = 0.7
            
            # Ground Truth histogram
            gt_channel_data = gt_intensities[channel] if isinstance(gt_intensities, dict) else []
            if len(gt_channel_data) > 0:
                ax_gt.hist(gt_channel_data, bins=bins, alpha=alpha, color=color, 
                          edgecolor='black', linewidth=0.1, density=True)
                
                # Statistics
                gt_mean = np.mean(gt_channel_data)
                gt_std = np.std(gt_channel_data)
                stats_text = f'Mean: {gt_mean:.1f}\nStd: {gt_std:.1f}\nPixels: {len(gt_channel_data)}'
                ax_gt.text(0.65, 0.85, stats_text, transform=ax_gt.transAxes, fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax_gt.set_title(f'GT - {channel} Channel', fontsize=12)
            ax_gt.set_xlim(0, 255)
            ax_gt.grid(True, alpha=0.3)
            if i == 0:
                ax_gt.set_ylabel('Density', fontsize=12)
            
            # Culled histogram
            culled_channel_data = culled_intensities[channel] if isinstance(culled_intensities, dict) else []
            if len(culled_channel_data) > 0:
                ax_culled.hist(culled_channel_data, bins=bins, alpha=alpha, color=color,
                              edgecolor='black', linewidth=0.1, density=True)
                
                # Statistics  
                culled_mean = np.mean(culled_channel_data)
                culled_std = np.std(culled_channel_data)
                stats_text = f'Mean: {culled_mean:.1f}\nStd: {culled_std:.1f}\nPixels: {len(culled_channel_data)}'
                ax_culled.text(0.65, 0.85, stats_text, transform=ax_culled.transAxes, fontsize=9,
                              bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            
            ax_culled.set_title(f'Culled - {channel} Channel', fontsize=12)
            ax_culled.set_xlabel('Pixel Intensity [0-255]', fontsize=12)
            ax_culled.set_xlim(0, 255)
            ax_culled.grid(True, alpha=0.3)
            if i == 0:
                ax_culled.set_ylabel('Density', fontsize=12)
        
        # Overall title
        method_name = "RGB Channels (Separate)"
        fig.suptitle(f'{method_name} Intensity Distribution Comparison (Masked Regions)\nPose {pose_id} - Pixel Threshold {pixel_threshold}', 
                    fontsize=16, fontweight='bold', y=0.95)
    
    else:
        # Single intensity histogram (luminance, mean, or flatten)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        
        # Common histogram parameters
        bins = np.arange(0, 257, 1)  # 0-255 with bin edges at 0.5, 1.5, ..., 255.5
        alpha = 0.7
        
        # Ground Truth histogram
        if len(gt_intensities) > 0:
            n_gt, bins_gt, patches_gt = ax1.hist(gt_intensities, bins=bins, alpha=alpha, 
                                                color='blue', edgecolor='black', linewidth=0.1, 
                                                density=True)
            
            # Calculate GT statistics
            gt_mean = np.mean(gt_intensities)
            gt_std = np.std(gt_intensities)
            gt_median = np.median(gt_intensities)
            gt_min = np.min(gt_intensities)
            gt_max = np.max(gt_intensities)
            
            # Add statistics text for GT
            pixel_count = len(gt_intensities) // 3 if method == "flatten" else len(gt_intensities)
            gt_stats_text = f'Mean: {gt_mean:.1f}\nStd: {gt_std:.1f}\nMedian: {gt_median:.1f}\nMin: {gt_min:.0f}\nMax: {gt_max:.0f}\nPixels: {pixel_count}'
            ax1.text(0.65, 0.85, gt_stats_text, transform=ax1.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            ax1.text(0.5, 0.5, 'No valid pixels in GT\n(no intensities to plot)', 
                    transform=ax1.transAxes, fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Method-specific titles
        method_names = {"luminance": "Luminance", "mean": "RGB Mean", "flatten": "All RGB Values"}
        method_name = method_names.get(method, method.title())
        
        ax1.set_title(f'Ground Truth (No Culling)', fontsize=14)
        ax1.set_xlabel(f'{method_name} Intensity [0-255]', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_xlim(0, 255)
        ax1.grid(True, alpha=0.3)
        
        # Culled image histogram
        if len(culled_intensities) > 0:
            n_culled, bins_culled, patches_culled = ax2.hist(culled_intensities, bins=bins, alpha=alpha, 
                                                            color='red', edgecolor='black', linewidth=0.1, 
                                                            density=True)
            
            # Calculate culled statistics
            culled_mean = np.mean(culled_intensities)
            culled_std = np.std(culled_intensities)
            culled_median = np.median(culled_intensities)
            culled_min = np.min(culled_intensities)
            culled_max = np.max(culled_intensities)
            
            # Add statistics text for culled
            pixel_count = len(culled_intensities) // 3 if method == "flatten" else len(culled_intensities)
            culled_stats_text = f'Mean: {culled_mean:.1f}\nStd: {culled_std:.1f}\nMedian: {culled_median:.1f}\nMin: {culled_min:.0f}\nMax: {culled_max:.0f}\nPixels: {pixel_count}'
            ax2.text(0.65, 0.85, culled_stats_text, transform=ax2.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'No valid pixels in culled\n(no intensities to plot)', 
                    transform=ax2.transAxes, fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        ax2.set_title(f'After Distance Culling (px≥{pixel_threshold})', fontsize=14)
        ax2.set_xlabel(f'{method_name} Intensity [0-255]', fontsize=12)
        ax2.set_xlim(0, 255)
        ax2.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(f'{method_name} Intensity Distribution Comparison (Masked Regions)\nPose {pose_id} - Pixel Threshold {pixel_threshold}', 
                    fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for suptitle
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved side-by-side histogram to: {save_path}")

def process_single_pose(image_path, pixel_threshold, save_dir, gt_masks_dir, method="luminance"):
    """
    Process a single pose image and create its side-by-side histogram plot.
    """
    # Extract pose ID from filename
    pose_id = int(re.search(r'pose_(\d+)\.png', os.path.basename(image_path)).group(1))
    print(f"Processing pose {pose_id}")
    
    try:
        # Load and extract GT and culled images
        gt_image, culled_image = load_stacked_image(image_path)
        
        # Load ground truth mask
        mask_path = os.path.join(gt_masks_dir, f"pose_{pose_id:06d}_mask.png")
        gt_mask = load_gt_mask(mask_path)
        
        if gt_mask is None:
            print(f"Skipping pose {pose_id} - no ground truth mask found")
            return
        
        print(f"GT mask coverage: {np.sum(gt_mask) / gt_mask.size * 100:.1f}%")
        
        # Extract pixel intensities from masked regions
        gt_intensities = extract_masked_pixel_intensities(gt_image, gt_mask, method=method)
        culled_intensities = extract_masked_pixel_intensities(culled_image, gt_mask, method=method)
        
        # Print pixel counts based on method
        if method == "separate":
            if isinstance(gt_intensities, dict):
                print(f"GT masked pixels per channel: {len(gt_intensities['R'])}")
                print(f"Culled masked pixels per channel: {len(culled_intensities['R'])}")
            else:
                print("Error: separate method should return dict")
        else:
            pixel_count_gt = len(gt_intensities) // 3 if method == "flatten" else len(gt_intensities)
            pixel_count_culled = len(culled_intensities) // 3 if method == "flatten" else len(culled_intensities)
            print(f"GT masked pixels: {pixel_count_gt}")
            print(f"Culled masked pixels: {pixel_count_culled}")
        
        # Create save path with method suffix
        save_path = os.path.join(save_dir, f"pose_{pose_id:06d}_intensity_histogram_{method}_comparison.png")
        
        # Create and save side-by-side histogram plot
        create_side_by_side_histogram(gt_intensities, culled_intensities, pose_id, pixel_threshold, save_path, method=method)
        
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
    render_height = 1422
    render_width = 2048
    # render_height = 356 # 1/4 of 1422
    # render_width = 512  # 1/4 of 2048
    
    # Intensity extraction method - CHOOSE ONE:
    # "luminance" - Proper luminance conversion (0.299*R + 0.587*G + 0.114*B) [RECOMMENDED]
    # "mean" - Simple RGB mean 
    # "flatten" - All RGB values flattened (original approach)
    # "separate" - Separate R, G, B histograms
    # intensity_method = "luminance"
    intensity_method = "mean"
    # intensity_method = "flatten"
    # intensity_method = "separate"
    
    print(f"{'='*80}")
    print(f"Processing pixel intensity histograms for {model_name}/{actor_name}/{seq_name}/{resolution}x/{frame_id}")
    print(f"Render resolution: {render_width}x{render_height}")
    print(f"Intensity method: {intensity_method}")
    print(f"{'='*80}")
    
    # Paths
    evaluation_images_dir = os.path.join(
        WORK_DIR, 
        f"results/{model_name}/{actor_name}/{seq_name}/resolution_{resolution}/{frame_id}/viewer_poses/render_hxw_{render_width}x{render_height}/static_dist_culling/evaluation_images"
    )
    
    gt_masks_dir = os.path.join(evaluation_images_dir, "gt_masks")
    
    save_dir = os.path.join(
        WORK_DIR, 
        f"scripts/plots/static_culling_pixel_intensity_analysis/{model_name}/{actor_name}_{seq_name}_{resolution}x_{frame_id}/render_hxw_{render_width}x{render_height}/{intensity_method}_method"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Reading evaluation images from: {evaluation_images_dir}")
    print(f"Reading GT masks from: {gt_masks_dir}")
    print(f"Saving histogram plots to: {save_dir}")
    
    # Check if GT masks directory exists
    if not os.path.exists(gt_masks_dir):
        print(f"Error: GT masks directory not found: {gt_masks_dir}")
        print("Please run evaluate_static_size_based_culling.py first to generate GT masks")
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
            process_single_pose(image_file, threshold, threshold_save_dir, gt_masks_dir, method=intensity_method)
    
    print(f"\n{'='*80}")
    print("Pixel intensity histogram analysis completed!")
    print(f"All histogram plots saved under: {save_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
