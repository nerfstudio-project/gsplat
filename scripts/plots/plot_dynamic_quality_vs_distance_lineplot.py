#!/usr/bin/env python3
"""
Plot comprehensive evaluation metrics for dynamic distance culling methods across different base pixel thresholds.
Generates plots for PSNR, SSIM, LPIPS, and their masked/cropped variants for dynamic culling.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from pathlib import Path

WORK_DIR = "/main/rajrup/Dropbox/Project/GsplatStream/gsplat"
FIG_SIZE = (8, 6)

def discover_base_pixel_thresholds(dynamic_culling_evaluation_dir):
    """
    Discover available base pixel thresholds from the detailed_results_dynamic_base_*.json files.
    
    Args:
        dynamic_culling_evaluation_dir: Path to the dynamic_culling_evaluation directory
        
    Returns:
        List of base pixel thresholds (as floats) sorted in ascending order
    """
    pattern = os.path.join(dynamic_culling_evaluation_dir, "detailed_results_dynamic_base_*.json")
    files = glob.glob(pattern)
    
    base_thresholds = []
    for file_path in files:
        match = re.search(r'detailed_results_dynamic_base_([0-9]+\.?[0-9]*)\.json', os.path.basename(file_path))
        if match:
            base_thresholds.append(float(match.group(1)))
    
    return sorted(base_thresholds)


def load_dynamic_metric_data(dynamic_culling_evaluation_dir, metric_name):
    """
    Load metric data for all base pixel thresholds and return sorted by distance.
    
    Args:
        dynamic_culling_evaluation_dir: Path to evaluation data
        metric_name: Name of the metric (e.g., "psnr", "masked_psnr", "ssim", etc.)
    
    Returns:
        dict: {base_threshold: {'distances': [...], 'values': [...], 'dynamic_thresholds': [...], 'color': ..., 'marker': ...}}
    """
    base_thresholds = discover_base_pixel_thresholds(dynamic_culling_evaluation_dir)
    if not base_thresholds:
        return {}
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(base_thresholds)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    all_data = {}
    for i, base_threshold in enumerate(base_thresholds):
        json_file = os.path.join(dynamic_culling_evaluation_dir, f"detailed_results_dynamic_base_{base_threshold}.json")
        if not os.path.exists(json_file):
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        distances = []
        values = []
        dynamic_thresholds = []
        
        for pose_id, metrics in data.items():
            distance = metrics["distance_to_bbox"]
            value = metrics[metric_name]
            dynamic_threshold = metrics["dynamic_pixel_threshold"]
            
            # Handle infinity values for PSNR metrics
            if metric_name in ["psnr", "masked_psnr"] and (value == float('inf') or str(value).lower() == 'infinity'):
                value = 100.0
            
            distances.append(distance)
            values.append(value)
            dynamic_thresholds.append(dynamic_threshold)
        
        # Sort by distance
        sorted_data = sorted(zip(distances, values, dynamic_thresholds))
        distances_sorted, values_sorted, dynamic_thresholds_sorted = zip(*sorted_data)
        
        all_data[base_threshold] = {
            'distances': distances_sorted,
            'values': values_sorted,
            'dynamic_thresholds': dynamic_thresholds_sorted,
            'color': colors[i],
            'marker': markers[i % len(markers)]
        }
    
    return all_data


def plot_dynamic_metric_comparison(dynamic_culling_evaluation_dir, save_dir, metric_name, metric_label, y_limit=None):
    """Create comparison plot for a specific metric across base pixel thresholds."""
    
    all_data = load_dynamic_metric_data(dynamic_culling_evaluation_dir, metric_name)
    if not all_data:
        print(f"No data found for {metric_name} in dynamic culling")
        return
    
    plt.figure(figsize=(10, 6))
    for base_threshold, data in all_data.items():
        plt.plot(data['distances'], data['values'], 
                color=data['color'], marker=data['marker'], 
                linewidth=2, markersize=4, alpha=0.7,
                label=f'Base Threshold {base_threshold}')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.title(f'{metric_label} vs Camera Distance - Dynamic Distance Culling', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if y_limit:
        plt.ylim(*y_limit)
    
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"{metric_name}_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"{metric_label} plot saved to: {plot_path}")
    plt.close()  # Close to free memory


def plot_dynamic_threshold_vs_distance(dynamic_culling_evaluation_dir, save_dir):
    """Plot Dynamic Pixel Threshold vs Distance for different base thresholds."""
    
    all_data = load_dynamic_metric_data(dynamic_culling_evaluation_dir, "psnr")  # Use any metric to get the data structure
    if not all_data:
        print("No data found for dynamic threshold plotting")
        return
    
    plt.figure(figsize=(10, 6))
    for base_threshold, data in all_data.items():
        plt.plot(data['distances'], data['dynamic_thresholds'], 
                color=data['color'], marker=data['marker'], 
                linewidth=2, markersize=4, alpha=0.7,
                label=f'Base Threshold {base_threshold}')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel('Dynamic Pixel Threshold', fontsize=12)
    plt.title('Dynamic Pixel Threshold vs Camera Distance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale since thresholds can vary significantly
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "dynamic_threshold_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Dynamic Threshold vs Distance plot saved to: {plot_path}")
    plt.close()


def plot_combined_dynamic_metrics(dynamic_culling_evaluation_dir, save_dir, metric_pairs):
    """Create combined plots with multiple metrics side by side."""
    
    for plot_name, (metric1, label1, metric2, label2) in metric_pairs.items():
        data1 = load_dynamic_metric_data(dynamic_culling_evaluation_dir, metric1)
        data2 = load_dynamic_metric_data(dynamic_culling_evaluation_dir, metric2)
        
        if not data1 or not data2:
            print(f"Missing data for combined plot {plot_name}")
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left subplot: First metric
        for base_threshold, data in data1.items():
            ax1.plot(data['distances'], data['values'], 
                    color=data['color'], marker=data['marker'], 
                    linewidth=2, markersize=4, alpha=0.7,
                    label=f'Base Threshold {base_threshold}')
        
        ax1.set_xlabel('Distance to Bounding Box Center (m)', fontsize=12)
        ax1.set_ylabel(label1, fontsize=12)
        ax1.set_title(f'{label1} vs Camera Distance', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Set appropriate y-limits based on metric type
        if metric1 in ["psnr", "masked_psnr"]:
            ax1.set_ylim(0.0, 80.0)
        elif metric1 in ["ssim", "masked_ssim"]:
            ax1.set_ylim(0.0, 1.0)
        
        ax1.legend()
        
        # Right subplot: Second metric
        for base_threshold, data in data2.items():
            ax2.plot(data['distances'], data['values'], 
                    color=data['color'], marker=data['marker'], 
                    linewidth=2, markersize=4, alpha=0.7,
                    label=f'Base Threshold {base_threshold}')
        
        ax2.set_xlabel('Distance to Bounding Box Center (m)', fontsize=12)
        ax2.set_ylabel(label2, fontsize=12)
        ax2.set_title(f'{label2} vs Camera Distance', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Set appropriate y-limits based on metric type
        if metric2 in ["psnr", "masked_psnr"]:
            ax2.set_ylim(0.0, 80.0)
        elif metric2 in ["ssim", "masked_ssim"]:
            ax2.set_ylim(0.0, 1.0)
        
        ax2.legend()
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, f"{plot_name}_combined.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Combined {plot_name} plot saved to: {plot_path}")
        plt.close()


def plot_dynamic_gaussians_vs_distance(dynamic_culling_evaluation_dir, save_dir):
    """Plot Number of Gaussians after dynamic culling vs Distance."""
    
    base_thresholds = discover_base_pixel_thresholds(dynamic_culling_evaluation_dir)
    if not base_thresholds:
        print("No base threshold data files found!")
        return
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(base_thresholds)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Collect data for all base thresholds
    all_data = {}
    total_gaussians = None  # Should be constant across all poses
    
    for i, base_threshold in enumerate(base_thresholds):
        json_file = os.path.join(dynamic_culling_evaluation_dir, f"detailed_results_dynamic_base_{base_threshold}.json")
        if not os.path.exists(json_file):
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        distances = []
        culled_gaussians = []
        wo_culled_gaussians = []
        
        for pose_id, metrics in data.items():
            distance = metrics["distance_to_bbox"]
            num_after_culling = metrics["culled_count"]
            wo_culled_count = metrics["wo_culled_count"]
            
            distances.append(distance)
            culled_gaussians.append(num_after_culling)
            wo_culled_gaussians.append(wo_culled_count)
        
        # Store total gaussians (should be same for all base thresholds)
        if total_gaussians is None:
            total_gaussians = wo_culled_gaussians[0]  # Take first value as reference
        
        # Sort by distance
        sorted_data = sorted(zip(distances, culled_gaussians))
        distances_sorted, culled_gaussians_sorted = zip(*sorted_data)
        
        all_data[base_threshold] = {
            'distances': distances_sorted,
            'culled_gaussians': culled_gaussians_sorted,
            'color': colors[i],
            'marker': markers[i % len(markers)]
        }
    
    # Create single plot for Gaussians vs Distance
    plt.figure(figsize=(10, 6))
    
    # Plot Number of Gaussians after culling vs Distance
    for base_threshold, data in all_data.items():
        plt.plot(data['distances'], data['culled_gaussians'], 
                color=data['color'], marker=data['marker'], 
                linewidth=2, markersize=4, alpha=0.7,
                label=f'Base Threshold {base_threshold}')
    
    # Add horizontal line for total gaussians (without culling)
    if total_gaussians is not None:
        plt.axhline(y=total_gaussians, color='red', linestyle='--', linewidth=2, 
                   alpha=0.8, label=f'Total Gaussians (no culling): {total_gaussians:,}')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel('Number of Gaussians after Dynamic Culling', fontsize=12)
    plt.title('Gaussians after Dynamic Culling vs Camera Distance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "gaussians_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gaussians vs Distance plot saved to: {plot_path}")
    plt.close()


def plot_dynamic_culling_ratio_vs_distance(dynamic_culling_evaluation_dir, save_dir):
    """Plot Culling Ratio vs Distance for dynamic culling."""
    
    all_data = load_dynamic_metric_data(dynamic_culling_evaluation_dir, "culling_ratio")
    if not all_data:
        print("No data found for culling ratio plotting")
        return
    
    plt.figure(figsize=(10, 6))
    for base_threshold, data in all_data.items():
        plt.plot(data['distances'], data['values'], 
                color=data['color'], marker=data['marker'], 
                linewidth=2, markersize=4, alpha=0.7,
                label=f'Base Threshold {base_threshold}')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel('Culling Ratio (%)', fontsize=12)
    plt.title('Dynamic Culling Ratio vs Camera Distance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)  # Culling ratio is always between 0 and 100%
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "culling_ratio_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Culling Ratio vs Distance plot saved to: {plot_path}")
    plt.close()


def plot_valid_pixels_vs_distance(dynamic_culling_evaluation_dir, save_dir):
    """Plot Number of Valid Pixels vs Distance (should be same across all base thresholds)."""
    
    base_thresholds = discover_base_pixel_thresholds(dynamic_culling_evaluation_dir)
    if not base_thresholds:
        print("No base threshold data files found!")
        return
    
    # Get valid pixels data (should be same across base thresholds, varies with pose)
    valid_pixels_data = None
    
    for base_threshold in base_thresholds:
        json_file = os.path.join(dynamic_culling_evaluation_dir, f"detailed_results_dynamic_base_{base_threshold}.json")
        if not os.path.exists(json_file):
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        distances = []
        valid_pixels = []
        
        for pose_id, metrics in data.items():
            distance = metrics["distance_to_bbox"]
            num_valid_pixels = metrics["num_valid_pixels"]
            
            distances.append(distance)
            valid_pixels.append(num_valid_pixels)
        
        # Sort by distance
        sorted_data = sorted(zip(distances, valid_pixels))
        distances_sorted, valid_pixels_sorted = zip(*sorted_data)
        
        # Store valid pixels data (should be same across base thresholds)
        if valid_pixels_data is None:
            valid_pixels_data = {
                'distances': distances_sorted,
                'valid_pixels': valid_pixels_sorted
            }
        break  # Only need one base threshold since valid pixels are the same across all
    
    # Create single plot for Valid Pixels vs Distance
    plt.figure(figsize=(10, 6))
    
    # Plot Number of Valid Pixels vs Distance
    if valid_pixels_data is not None:
        plt.plot(valid_pixels_data['distances'], valid_pixels_data['valid_pixels'], 
                color='blue', marker='o', linewidth=2, markersize=4, alpha=0.7,
                label='Valid Pixels (same for all base thresholds)')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel('Number of Valid Pixels (log scale)', fontsize=12)
    plt.title('Valid Pixels vs Camera Distance - Dynamic Distance Culling', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "valid_pixels_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Valid Pixels vs Distance plot saved to: {plot_path}")
    plt.close()


def generate_all_dynamic_plots(dynamic_culling_evaluation_dir, save_dir):
    """Generate all plots for dynamic distance culling."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating dynamic culling plots in {save_dir}")
    
    # Define all metrics to plot individually
    individual_metrics = [
        ("psnr", "PSNR (dB)", (0.0, 80.0)),
        ("masked_psnr", "Masked PSNR (dB)", (0.0, 80.0)),
        ("ssim", "SSIM", (0.0, 1.0)),
        ("masked_ssim", "Masked SSIM", (0.0, 1.0)),
        ("lpips", "LPIPS", None),
        ("cropped_lpips", "Cropped LPIPS", None),
        ("masked_se", "Masked SE", None),
        ("masked_rmse", "Masked RMSE", None)
    ]
    
    # Generate individual metric plots
    for metric_name, metric_label, y_limit in individual_metrics:
        plot_dynamic_metric_comparison(dynamic_culling_evaluation_dir, save_dir, 
                                     metric_name, metric_label, y_limit)
    
    # Define combined plots (side-by-side comparisons)
    combined_plots = {
        "psnr_comparison": ("psnr", "PSNR (dB)", "masked_psnr", "Masked PSNR (dB)"),
        "ssim_comparison": ("ssim", "SSIM", "masked_ssim", "Masked SSIM"),
        "lpips_comparison": ("lpips", "LPIPS", "cropped_lpips", "Cropped LPIPS"),
        "error_comparison": ("masked_se", "Masked SE", "masked_rmse", "Masked RMSE")
    }
    
    # Generate combined plots
    plot_combined_dynamic_metrics(dynamic_culling_evaluation_dir, save_dir, combined_plots)
    
    # Generate special analysis plots
    plot_dynamic_threshold_vs_distance(dynamic_culling_evaluation_dir, save_dir)
    plot_dynamic_gaussians_vs_distance(dynamic_culling_evaluation_dir, save_dir)
    plot_dynamic_culling_ratio_vs_distance(dynamic_culling_evaluation_dir, save_dir)
    plot_valid_pixels_vs_distance(dynamic_culling_evaluation_dir, save_dir)


if __name__ == "__main__":

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
    print(f"Plotting Dynamic Culling Quality vs Distance for {model_name}/{actor_name}/{seq_name}/{resolution}x/{frame_id} with render resolution {render_width}x{render_height}")
    print(f"{'='*80}")

    save_dir = os.path.join(WORK_DIR, f"scripts/plots/dynamic_culling_quality_vs_distance_line_plots/{model_name}/{actor_name}_{seq_name}_{resolution}x_{frame_id}", f"render_hxw_{render_width}x{render_height}")
    print(f"Plots will be saved to: {save_dir}")
    
    # Path to the dynamic culling evaluation directory
    dynamic_culling_evaluation_dir = os.path.join(WORK_DIR, f"results/{model_name}/{actor_name}/{seq_name}/resolution_{resolution}/{frame_id}/viewer_poses/render_hxw_{render_width}x{render_height}/dynamic_culling_evaluation")
    
    # Check if directory exists
    if not os.path.exists(dynamic_culling_evaluation_dir):
        print(f"Error: Dynamic culling evaluation directory not found: {dynamic_culling_evaluation_dir}")
        print("Please run the dynamic culling evaluation script first.")
        exit(1)
    
    # Generate all plots for dynamic distance culling
    print(f"\n{'='*80}")
    print("Processing Dynamic Distance Culling")
    print(f"{'='*80}")
    
    generate_all_dynamic_plots(dynamic_culling_evaluation_dir, save_dir)
    
    print(f"\nCompleted plots for dynamic distance culling")
    
    print(f"\n{'='*80}")
    print("All dynamic culling plotting completed!")
    print(f"All plots saved under: {save_dir}")
    print(f"{'='*80}")
