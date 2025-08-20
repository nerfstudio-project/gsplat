#!/usr/bin/env python3
"""
Plot histograms of pixel coverage for each pose from saved .npy files.
This script reads pixel size data saved by evaluate_quality_of_culling.py
and creates individual histogram plots for each pose.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path

WORK_DIR = "/ssd1/rajrup/Project/gsplat"

def load_poses(poses_file: str) -> list:
    """Load viewer poses from JSON file."""
    if not os.path.exists(poses_file):
        raise FileNotFoundError(f"Poses file not found: {poses_file}")
    
    with open(poses_file, 'r') as f:
        poses = json.load(f)
    
    print(f"Loaded {len(poses)} poses from {poses_file}")
    return poses

def load_pixel_sizes(pixel_sizes_file: str) -> np.ndarray:
    """Load pixel sizes from .npy file."""
    if not os.path.exists(pixel_sizes_file):
        raise FileNotFoundError(f"Pixel sizes file not found: {pixel_sizes_file}")
    
    pixel_sizes = np.load(pixel_sizes_file)
    print(f"Loaded {len(pixel_sizes)} pixel sizes from {pixel_sizes_file}")
    return pixel_sizes

def plot_pixel_size_histogram(pixel_sizes: np.ndarray, pose_id: int, pose_data: dict, save_path: str, 
                             bin_width: float = 0.5, x_limit: tuple = None, log_scale: bool = True,
                             render_width: int = 2048, render_height: int = 1422):
    """Plot histogram of pixel sizes for a single pose."""
    
    # Calculate statistics
    mean_size = np.mean(pixel_sizes)
    median_size = np.median(pixel_sizes)
    min_size = np.min(pixel_sizes)
    max_size = np.max(pixel_sizes)
    std_size = np.std(pixel_sizes)
    distance = pose_data.get('distance_to_bbox_center', 0.0)
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Create bins with specified width
    if x_limit is not None:
        x_min, x_max = x_limit
        bins = np.arange(x_min, x_max + bin_width, bin_width)
    else:
        bins = np.arange(0, max_size + bin_width, bin_width)
    
    # Plot histogram
    counts, bin_edges, patches = plt.hist(pixel_sizes, bins=bins, alpha=0.7, color='skyblue', 
                                         edgecolor='black', linewidth=0.5, range=x_limit)
    
    # Add vertical lines for statistics
    plt.axvline(mean_size, color='red', linestyle='--', linewidth=2, label=f'Mean (px): {mean_size:.2f}')
    plt.axvline(median_size, color='green', linestyle='--', linewidth=2, label=f'Median (px): {median_size:.2f}')
    
    # Add vertical line for common thresholds
    common_thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
    colors = ['orange', 'purple', 'brown', 'pink', 'gray']
    for threshold, color in zip(common_thresholds, colors):
        if min_size <= threshold <= max_size:
            plt.axvline(threshold, color=color, linestyle=':', linewidth=1.5, alpha=0.8, 
                       label=f'Threshold (px): {threshold}')
    
    # Set labels and title
    plt.xlabel('Pixel Coverage (px)', fontsize=12)
    plt.ylabel('Number of Gaussians', fontsize=12)
    plt.title(f'Pixel Coverage Distribution - Pose {pose_id}, '
              f'Distance: {distance:.3f}m, Render Resolution: {render_width}x{render_height}', fontsize=14)
    
    # Set x-axis limits if specified
    if x_limit:
        plt.xlim(x_limit)
    
    # Set y-axis to log scale if requested
    if log_scale:
        plt.yscale('log')
        plt.ylabel('Number of Gaussians (log scale)', fontsize=12)
    plt.ylim(10, 1e6)
    
    # Add statistics text box
    stats_text = (f'Statistics:\n'
                  f'# GS: {len(pixel_sizes):,}\n'
                  f'Mean (px): {mean_size:.3f}\n'
                  f'Median (px): {median_size:.3f}\n'
                  f'Std (px): {std_size:.3f}\n'
                  f'Min (px): {min_size:.3f}\n'
                  f'Max (px): {max_size:.3f}')
    
    plt.text(0.80, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10, fontfamily='monospace')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', bbox_to_anchor=(0.97, 0.75))
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to: {save_path}")
    plt.close()

def plot_summary_statistics(pixel_sizes_dir: str, poses: list, save_dir: str):
    """Plot summary statistics across all poses."""
    
    pose_ids = []
    distances = []
    mean_pixel_sizes = []
    median_pixel_sizes = []
    gaussian_counts = []
    
    for pose in poses:
        pose_id = pose['pose_id']
        distance = pose.get('distance_to_bbox_center', 0.0)
        
        pixel_sizes_file = os.path.join(pixel_sizes_dir, f"{pose_id}.npy")
        if os.path.exists(pixel_sizes_file):
            pixel_sizes = np.load(pixel_sizes_file)
            
            pose_ids.append(pose_id)
            distances.append(distance)
            mean_pixel_sizes.append(np.mean(pixel_sizes))
            median_pixel_sizes.append(np.median(pixel_sizes))
            gaussian_counts.append(len(pixel_sizes))
    
    if not pose_ids:
        print("No pixel size data found for summary plots")
        return
    
    # Create subplots for summary statistics
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 6))
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Mean pixel size vs distance
    ax1.scatter(distances, mean_pixel_sizes, alpha=0.7, s=30)
    ax1.set_xlabel('Distance to Bounding Box Center (m)')
    ax1.set_ylabel('Mean Pixel Size')
    ax1.set_title('Mean Pixel Size vs Camera Distance')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Median pixel size vs distance
    ax2.scatter(distances, median_pixel_sizes, alpha=0.7, s=30, color='orange')
    ax2.set_xlabel('Distance to Bounding Box Center (m)')
    ax2.set_ylabel('Median Pixel Size')
    ax2.set_title('Median Pixel Size vs Camera Distance')
    ax2.grid(True, alpha=0.3)
    
    # # Plot 3: Number of Gaussians vs distance
    # ax3.scatter(distances, gaussian_counts, alpha=0.7, s=30, color='green')
    # ax3.set_xlabel('Distance to Bounding Box Center (m)')
    # ax3.set_ylabel('Number of Gaussians')
    # ax3.set_title('Number of Gaussians vs Camera Distance')
    # ax3.grid(True, alpha=0.3)
    
    # # Plot 4: Mean vs Median pixel size
    # ax4.scatter(mean_pixel_sizes, median_pixel_sizes, alpha=0.7, s=30, color='red')
    # ax4.set_xlabel('Mean Pixel Size')
    # ax4.set_ylabel('Median Pixel Size')
    # ax4.set_title('Mean vs Median Pixel Size')
    # # Add diagonal line for reference
    # min_val = min(min(mean_pixel_sizes), min(median_pixel_sizes))
    # max_val = max(max(mean_pixel_sizes), max(median_pixel_sizes))
    # ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    # ax4.legend()
    # ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = os.path.join(save_dir, "pixel_size_summary_statistics.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Summary statistics plot saved to: {summary_path}")
    plt.close()

def plot_overall_histogram(pixel_sizes_dir: str, poses: list, save_dir: str,
                          bin_width: float = 0.5, x_limit: tuple = None, log_scale: bool = True):
    """Plot overall histogram combining all poses."""
    
    all_pixel_sizes = []
    total_gaussians = 0
    
    for pose in poses:
        pose_id = pose['pose_id']
        pixel_sizes_file = os.path.join(pixel_sizes_dir, f"{pose_id}.npy")
        
        if os.path.exists(pixel_sizes_file):
            pixel_sizes = np.load(pixel_sizes_file)
            all_pixel_sizes.extend(pixel_sizes)
            total_gaussians += len(pixel_sizes)
    
    if not all_pixel_sizes:
        print("No pixel size data found for overall histogram")
        return
    
    all_pixel_sizes = np.array(all_pixel_sizes)
    
    # Calculate overall statistics
    mean_size = np.mean(all_pixel_sizes)
    median_size = np.median(all_pixel_sizes)
    min_size = np.min(all_pixel_sizes)
    max_size = np.max(all_pixel_sizes)
    std_size = np.std(all_pixel_sizes)
    
    # Create overall histogram
    plt.figure(figsize=(14, 10))
    
    # Create bins with specified width
    if x_limit is not None:
        x_min, x_max = x_limit
        bins = np.arange(x_min, x_max + bin_width, bin_width)
    else:
        bins = np.arange(0, max_size + bin_width, bin_width)
    
    # Plot histogram
    counts, bin_edges, patches = plt.hist(all_pixel_sizes, bins=bins, alpha=0.7, color='lightcoral', 
                                         edgecolor='black', linewidth=0.5, range=x_limit)
    
    # Add vertical lines for statistics
    plt.axvline(mean_size, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_size:.2f}')
    plt.axvline(median_size, color='green', linestyle='--', linewidth=2, label=f'Median: {median_size:.2f}')
    
    # Add vertical lines for common thresholds
    common_thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
    colors = ['orange', 'purple', 'brown', 'pink', 'gray']
    for threshold, color in zip(common_thresholds, colors):
        if min_size <= threshold <= max_size:
            plt.axvline(threshold, color=color, linestyle=':', linewidth=1.5, alpha=0.8, 
                       label=f'Threshold: {threshold}')
    
    # Set labels and title
    plt.xlabel('Pixel Coverage (px)', fontsize=12)
    plt.ylabel('Number of Gaussians', fontsize=12)
    plt.title(f'Overall Pixel Size Distribution\n'
              f'Total Gaussians: {total_gaussians:,} from {len(poses)} poses', fontsize=14)
    
    # Set x-axis limits if specified
    if x_limit:
        plt.xlim(x_limit)
    
    # Set y-axis to log scale if requested
    if log_scale:
        plt.yscale('log')
        plt.ylabel('Number of Gaussians (log scale)', fontsize=12)
    
    # Add statistics text box
    stats_text = (f'Overall Statistics:\n'
                  f'Total Count: {len(all_pixel_sizes):,}\n'
                  f'Poses: {len(poses)}\n'
                  f'Mean (px): {mean_size:.3f}\n'
                  f'Median (px): {median_size:.3f}\n'
                  f'Std (px): {std_size:.3f}\n'
                  f'Min (px): {min_size:.3f}\n'
                  f'Max (px): {max_size:.3f}')
    
    plt.text(0.80, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10, fontfamily='monospace')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', bbox_to_anchor=(0.97, 0.75))
    plt.tight_layout()
    
    # Save overall histogram
    overall_path = os.path.join(save_dir, "pixel_size_overall_histogram.png")
    plt.savefig(overall_path, dpi=300, bbox_inches='tight')
    print(f"Overall histogram saved to: {overall_path}")
    plt.close()

def main():
    """Main function to generate pixel size histograms."""
    
    # Configuration - match the structure from plot_psnr_vs_distance.py
    model_name = "actorshq_l1_0.5_ssim_0.5_alpha_1.0"
    actor_name = "Actor01"
    seq_name = "Sequence1"
    resolution = 4
    frame_id = 0
    render_height = 1422   # 1422
    render_width = 2048    # 2048
    # render_height = 711   # 1/2 of 1422
    # render_width = 1024   # 1/2 of 2048
    # render_height = 356   # 1/4 of 1422
    # render_width = 512   # 1/4 of 2048

    print(f"{'='*80}")
    print(f"Plotting pixel size histograms for {model_name}/{actor_name}/{seq_name}/{resolution}x/{frame_id} with render resolution {render_width}x{render_height}")
    print(f"{'='*80}")

    # Base directories
    base_result_dir = os.path.join(WORK_DIR, f"results/{model_name}/{actor_name}/{seq_name}/resolution_{resolution}/{frame_id}")
    poses_dir = os.path.join(base_result_dir, "viewer_poses")
    pixel_sizes_dir = os.path.join(poses_dir, f"render_hxw_{render_width}x{render_height}", "static_dist_culling/pixel_sizes")
    
    # Output directory for histograms
    base_save_dir = os.path.join(WORK_DIR, f"scripts/plots/static_culling_pixel_size_histograms/{model_name}/{actor_name}_{seq_name}_{resolution}x_{frame_id}", f"render_hxw_{render_width}x{render_height}")
    os.makedirs(base_save_dir, exist_ok=True)
    
    print(f"Base result directory: {base_result_dir}")
    print(f"Pixel sizes directory: {pixel_sizes_dir}")
    print(f"Histograms will be saved to: {base_save_dir}")
    
    # Check if directories exist
    if not os.path.exists(poses_dir):
        raise FileNotFoundError(f"Poses directory not found: {poses_dir}")
    
    if not os.path.exists(pixel_sizes_dir):
        raise FileNotFoundError(f"Pixel sizes directory not found: {pixel_sizes_dir}")
    
    # Load poses
    poses_file = os.path.join(poses_dir, "viewer_poses.json")
    poses = load_poses(poses_file)
    
    # # Filter for highest resolution if requested
    # if len(poses) > 0:
    #     max_width = max(pose["width"] for pose in poses)
    #     max_height = max(pose["height"] for pose in poses)
    #     poses = [pose for pose in poses if pose["width"] == max_width and pose["height"] == max_height]
    #     print(f"Filtered to {len(poses)} poses at highest resolution ({max_width}x{max_height})")
    
    if len(poses) == 0:
        raise ValueError("No poses found for histogram generation")
    
    # Create individual histogram subdirectory
    individual_histograms_dir = base_save_dir
    os.makedirs(individual_histograms_dir, exist_ok=True)
    
    # Determine consistent x-axis limits across all poses
    print(f"\nDetermining consistent x-axis limits across all poses...")
    all_pixel_sizes_for_limit = []
    
    for pose in poses:
        pose_id = pose['pose_id']
        pixel_sizes_file = os.path.join(pixel_sizes_dir, f"{pose_id}.npy")
        
        if os.path.exists(pixel_sizes_file):
            pixel_sizes = np.load(pixel_sizes_file)
            all_pixel_sizes_for_limit.extend(pixel_sizes)
    
    if all_pixel_sizes_for_limit:
        all_pixel_sizes_array = np.array(all_pixel_sizes_for_limit)
        # Use 99th percentile or 30, whichever is smaller, as the upper limit
        # x_max = min(50.0, np.percentile(all_pixel_sizes_array, 99))
        x_max = 40.0
        x_limit = (0, x_max)
        print(f"Using consistent x-axis limit: {x_limit}")
    else:
        x_limit = (0, 30)
        print(f"No data found, using default x-axis limit: {x_limit}")
    
    # Generate histogram for each pose
    print(f"\nGenerating individual histograms for {len(poses)} poses...")
    generated_count = 0
    
    for pose in poses:
        pose_id = pose['pose_id']
        pixel_sizes_file = os.path.join(pixel_sizes_dir, f"{pose_id}.npy")
        
        if os.path.exists(pixel_sizes_file):
            try:
                pixel_sizes = load_pixel_sizes(pixel_sizes_file)
                
                histogram_path = os.path.join(individual_histograms_dir, f"pose_{pose_id}.png")
                plot_pixel_size_histogram(pixel_sizes, pose_id, pose, histogram_path, 
                                         bin_width=0.5, x_limit=x_limit, log_scale=True,
                                         render_width=render_width, render_height=render_height)
                generated_count += 1
                
            except Exception as e:
                print(f"Error generating histogram for pose {pose_id}: {e}")
        else:
            print(f"Warning: Pixel sizes file not found for pose {pose_id}: {pixel_sizes_file}")
    
    print(f"Generated {generated_count} individual histograms")
    
    # Generate summary statistics plot
    print("\nGenerating summary statistics plot...")
    try:
        plot_summary_statistics(pixel_sizes_dir, poses, base_save_dir)
    except Exception as e:
        print(f"Error generating summary statistics: {e}")
    
    # # Generate overall histogram
    # print("\nGenerating overall histogram...")
    # try:
    #     plot_overall_histogram(pixel_sizes_dir, poses, base_save_dir, 
    #                           bin_width=0.5, x_limit=x_limit, log_scale=True)
    # except Exception as e:
    #     print(f"Error generating overall histogram: {e}")
    
    print(f"\n{'='*80}")
    print("Pixel size histogram generation completed!")
    print(f"All plots saved under: {base_save_dir}")
    print(f"Individual histograms: {individual_histograms_dir}")
    print(f"Generated {generated_count} individual histograms")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
