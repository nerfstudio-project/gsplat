#!/usr/bin/env python3
"""
Plot comprehensive evaluation metrics for Gaussian merging methods across different thresholds.
Generates plots for PSNR, SSIM, LPIPS, and their masked/cropped variants for merging evaluation.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from pathlib import Path

WORK_DIR = "/ssd1/rajrup/Project/gsplat"
FIG_SIZE = (8, 6)

def discover_merging_methods(merging_evaluation_dir):
    """
    Discover available merging methods from the directory structure.
    
    Args:
        merging_evaluation_dir: Path to the merging_evaluation directory
        
    Returns:
        List of method names (e.g., ['merge_center_in_pixel_weighted_px', 'merge_center_in_pixel_weighted_area'])
    """
    if not os.path.exists(merging_evaluation_dir):
        return []
    
    methods = []
    for item in os.listdir(merging_evaluation_dir):
        item_path = os.path.join(merging_evaluation_dir, item)
        if os.path.isdir(item_path) and item.startswith('merge_'):
            methods.append(item)
    
    return sorted(methods)


def discover_pixel_thresholds_for_method(merging_evaluation_dir, method_name):
    """
    Discover available pixel thresholds for a specific merging method.
    
    Args:
        merging_evaluation_dir: Path to the merging_evaluation directory
        method_name: Name of the merging method
        
    Returns:
        List of all pixel thresholds found across all parameter configurations
    """
    method_dir = os.path.join(merging_evaluation_dir, method_name)
    if not os.path.exists(method_dir):
        return []
    
    all_pixel_thresholds = []
    
    # Look for parameter configuration subdirectories
    param_dirs = sorted(os.listdir(method_dir))
    for param_dir in param_dirs:
        param_path = os.path.join(method_dir, param_dir)
        if os.path.isdir(param_path):
            # Look for detailed_results_*.json files in this parameter directory
            pattern = os.path.join(param_path, "detailed_results_*.json")
            files = glob.glob(pattern)
            
            for file_path in files:
                match = re.search(r'detailed_results_([0-9]+\.?[0-9]*)\.json', os.path.basename(file_path))
                if match:
                    all_pixel_thresholds.append(float(match.group(1)))
    
    # Return unique, sorted pixel thresholds
    return sorted(list(set(all_pixel_thresholds)))


def load_method_data(merging_evaluation_dir, method_name, metric_name):
    """
    Load metric data for a specific merging method across all pixel thresholds.
    
    Args:
        merging_evaluation_dir: Path to evaluation data
        method_name: Method name (e.g., "merge_center_in_pixel_weighted_px")
        metric_name: Name of the metric (e.g., "psnr", "masked_psnr", "ssim", etc.)
    
    Returns:
        dict: {pixel_threshold: {'distances': [...], 'values': [...], 'color': ..., 'marker': ...}}
    """
    pixel_thresholds = discover_pixel_thresholds_for_method(merging_evaluation_dir, method_name)
    if not pixel_thresholds:
        return {}
    
    print(f"Found pixel thresholds for {method_name}: {pixel_thresholds}")
    
    # Assign colors and markers based on sorted pixel thresholds
    colors = plt.cm.viridis(np.linspace(0, 1, len(pixel_thresholds)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    method_data = {}
    
    # Find the parameter configuration directory that contains our data
    method_dir = os.path.join(merging_evaluation_dir, method_name)
    param_dirs = sorted(os.listdir(method_dir))
    
    for i, pixel_threshold in enumerate(pixel_thresholds):
        # Find which parameter directory contains this pixel threshold
        found_data = False
        for param_dir in param_dirs:
            param_path = os.path.join(method_dir, param_dir)
            if os.path.isdir(param_path):
                json_file = os.path.join(param_path, f"detailed_results_{pixel_threshold}.json")
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # The data should be in format: {"parameters": {...}, "param_str": "...", "results": {pose_id: {...}}}
                    if "results" not in data:
                        print(f"No results found in {json_file}")
                        continue
                    
                    pose_results = data["results"]
                    
                    distances = []
                    values = []
                    
                    for pose_id, metrics in pose_results.items():
                        distance = metrics["distance_to_bbox"]
                        value = metrics[metric_name]
                        
                        # Handle infinity values for PSNR metrics
                        if metric_name in ["psnr", "masked_psnr"] and (value == float('inf') or str(value).lower() == 'infinity'):
                            value = 100.0
                        
                        distances.append(distance)
                        values.append(value)
                    
                    # Sort by distance
                    sorted_data = sorted(zip(distances, values))
                    distances_sorted, values_sorted = zip(*sorted_data) if sorted_data else ([], [])
                    
                    method_data[pixel_threshold] = {
                        'distances': distances_sorted,
                        'values': values_sorted,
                        'color': colors[i],
                        'marker': markers[i % len(markers)]
                    }
                    
                    found_data = True
                    break
        
        if not found_data:
            print(f"Warning: Could not find data for pixel threshold {pixel_threshold}")
    
    return method_data


def plot_method_metric_comparison(merging_evaluation_dir, save_dir, method_name, metric_name, metric_label, y_limit=None):
    """Create comparison plot for a specific metric across pixel thresholds for a merging method."""
    
    method_data = load_method_data(merging_evaluation_dir, method_name, metric_name)
    if not method_data:
        print(f"No data found for {metric_name} in {method_name}")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot each pixel threshold
    for pixel_threshold, data in method_data.items():
        label = f'Pixel Threshold {pixel_threshold}'
        plt.plot(data['distances'], data['values'], 
                color=data['color'], marker=data['marker'], 
                linewidth=2, markersize=4, alpha=0.7,
                label=label)
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.title(f'{metric_label} vs Camera Distance - {method_name.replace("_", " ").title()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if y_limit:
        plt.ylim(*y_limit)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"{metric_name}_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"{metric_label} plot saved to: {plot_path}")
    plt.close()  # Close to free memory


def plot_method_combined_metrics(merging_evaluation_dir, save_dir, method_name, metric_pairs):
    """Create combined plots with multiple metrics side by side for a merging method."""
    
    for plot_name, (metric1, label1, metric2, label2) in metric_pairs.items():
        data1 = load_method_data(merging_evaluation_dir, method_name, metric1)
        data2 = load_method_data(merging_evaluation_dir, method_name, metric2)
        
        if not data1 or not data2:
            print(f"Missing data for combined plot {plot_name} in {method_name}")
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left subplot: First metric
        for pixel_threshold, data in data1.items():
            label = f'Pixel Threshold {pixel_threshold}'
            ax1.plot(data['distances'], data['values'], 
                    color=data['color'], marker=data['marker'], 
                    linewidth=2, markersize=4, alpha=0.7,
                    label=label)
        
        ax1.set_xlabel('Distance to Bounding Box Center (m)', fontsize=12)
        ax1.set_ylabel(label1, fontsize=12)
        ax1.set_title(f'{label1} vs Camera Distance', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Set appropriate y-limits based on metric type
        if metric1 in ["psnr", "masked_psnr"]:
            ax1.set_ylim(0.0, 80.0)
        elif metric1 in ["ssim", "masked_ssim"]:
            ax1.set_ylim(0.0, 1.0)
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Right subplot: Second metric
        for pixel_threshold, data in data2.items():
            label = f'Pixel Threshold {pixel_threshold}'
            ax2.plot(data['distances'], data['values'], 
                    color=data['color'], marker=data['marker'], 
                    linewidth=2, markersize=4, alpha=0.7,
                    label=label)
        
        ax2.set_xlabel('Distance to Bounding Box Center (m)', fontsize=12)
        ax2.set_ylabel(label2, fontsize=12)
        ax2.set_title(f'{label2} vs Camera Distance', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Set appropriate y-limits based on metric type
        if metric2 in ["psnr", "masked_psnr"]:
            ax2.set_ylim(0.0, 80.0)
        elif metric2 in ["ssim", "masked_ssim"]:
            ax2.set_ylim(0.0, 1.0)
        
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, f"{plot_name}_combined.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Combined {plot_name} plot saved to: {plot_path}")
        plt.close()


def plot_method_gaussians_vs_distance(merging_evaluation_dir, save_dir, method_name):
    """Plot Number of Gaussians after merging vs Distance for a specific method."""
    
    method_data = load_method_data(merging_evaluation_dir, method_name, "processed_count")  # Use processed_count metric
    if not method_data:
        print(f"No data found for Gaussians count in {method_name}")
        return
    
    plt.figure(figsize=(12, 8))
    total_gaussians = None  # Should be constant across all poses
    
    # Plot each pixel threshold
    for pixel_threshold, data in method_data.items():
        label = f'Pixel Threshold {pixel_threshold}'
        plt.plot(data['distances'], data['values'], 
                color=data['color'], marker=data['marker'], 
                linewidth=2, markersize=4, alpha=0.7,
                label=label)
        
        # Get original count from the same data files for the horizontal line
        if total_gaussians is None:
            # Load original_count from the same method data
            original_data = load_method_data(merging_evaluation_dir, method_name, "original_count")
            if original_data:
                # Get first available original count (should be same for all)
                for _, o_data in original_data.items():
                    if o_data['values']:
                        total_gaussians = o_data['values'][0]
                        break
    
    # Add horizontal line for total gaussians (without merging)
    if total_gaussians is not None:
        plt.axhline(y=total_gaussians, color='red', linestyle='--', linewidth=2, 
                   alpha=0.8, label=f'Total Gaussians (no merging): {total_gaussians:,}')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel('Number of Gaussians after Merging', fontsize=12)
    plt.title(f'Gaussians after Merging vs Camera Distance - {method_name.replace("_", " ").title()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.ylim(bottom=0)
    
    plot_path = os.path.join(save_dir, "gaussians_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gaussians vs Distance plot saved to: {plot_path}")
    plt.close()


def plot_method_reduction_ratio_vs_distance(merging_evaluation_dir, save_dir, method_name):
    """Plot Merging Reduction Ratio vs Distance for a specific method."""
    
    method_data = load_method_data(merging_evaluation_dir, method_name, "processing_ratio")
    if not method_data:
        print(f"No data found for reduction ratio in {method_name}")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot each pixel threshold
    for pixel_threshold, data in method_data.items():
        label = f'Pixel Threshold {pixel_threshold}'
        plt.plot(data['distances'], data['values'], 
                color=data['color'], marker=data['marker'], 
                linewidth=2, markersize=4, alpha=0.7,
                label=label)
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel('Merging Reduction Ratio (%)', fontsize=12)
    plt.title(f'Merging Reduction Ratio vs Camera Distance - {method_name.replace("_", " ").title()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "reduction_ratio_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Reduction Ratio vs Distance plot saved to: {plot_path}")
    plt.close()


def plot_method_valid_pixels_vs_distance(merging_evaluation_dir, save_dir, method_name):
    """Plot Number of Valid Pixels vs Distance for a specific method."""
    
    method_data = load_method_data(merging_evaluation_dir, method_name, "num_valid_pixels")
    if not method_data:
        print(f"No data found for valid pixels in {method_name}")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot only one pixel threshold since valid pixels should be same across all
    # (they depend on the pose, not on the merging method)
    first_pixel_threshold = next(iter(method_data.keys()))
    data = method_data[first_pixel_threshold]
    
    plt.plot(data['distances'], data['values'], 
            color='blue', marker='o', linewidth=2, markersize=4, alpha=0.7,
            label='Valid Pixels (same for all configurations)')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel('Number of Valid Pixels (log scale)', fontsize=12)
    plt.title(f'Valid Pixels vs Camera Distance - {method_name.replace("_", " ").title()}', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "valid_pixels_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Valid Pixels vs Distance plot saved to: {plot_path}")
    plt.close()


def generate_all_plots_for_method(merging_evaluation_dir, base_save_dir, method_name):
    """Generate all plots for a specific merging method."""
    
    # Create method-specific subdirectory
    method_save_dir = os.path.join(base_save_dir, method_name)
    os.makedirs(method_save_dir, exist_ok=True)
    
    print(f"Generating plots for {method_name.replace('_', ' ').title()} in {method_save_dir}")
    
    # Define all metrics to plot individually
    individual_metrics = [
        ("psnr", "PSNR (dB)", (0.0, 80.0)),
        ("masked_psnr", "Masked PSNR (dB)", (0.0, 80.0)),
        ("ssim", "SSIM", (0.0, 1.0)),
        ("masked_ssim", "Masked SSIM", (0.0, 1.0)),
        ("lpips", "LPIPS", None),
        ("cropped_lpips", "Cropped LPIPS", None)
    ]
    
    # Generate individual metric plots
    for metric_name, metric_label, y_limit in individual_metrics:
        plot_method_metric_comparison(merging_evaluation_dir, method_save_dir, method_name, 
                             metric_name, metric_label, y_limit)
    
    # Define combined plots (side-by-side comparisons)
    combined_plots = {
        "psnr_comparison": ("psnr", "PSNR (dB)", "masked_psnr", "Masked PSNR (dB)"),
        "ssim_comparison": ("ssim", "SSIM", "masked_ssim", "Masked SSIM"),
        "lpips_comparison": ("lpips", "LPIPS", "cropped_lpips", "Cropped LPIPS")
    }
    
    # Generate combined plots
    plot_method_combined_metrics(merging_evaluation_dir, method_save_dir, method_name, combined_plots)
    
    # Generate special analysis plots
    plot_method_gaussians_vs_distance(merging_evaluation_dir, method_save_dir, method_name)
    plot_method_reduction_ratio_vs_distance(merging_evaluation_dir, method_save_dir, method_name)
    plot_method_valid_pixels_vs_distance(merging_evaluation_dir, method_save_dir, method_name)





if __name__ == "__main__":

    model_name = "actorshq_l1_0.5_ssim_0.5_alpha_1.0"
    # actor_name = "Actor01"
    actor_name = "Actor08"
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
    print(f"Plotting Merging Quality vs Distance for {model_name}/{actor_name}/{seq_name}/{resolution}x/{frame_id} with render resolution {render_width}x{render_height}")
    print(f"{'='*80}")

    base_save_dir = os.path.join(WORK_DIR, f"scripts/plots/static_merging_quality_vs_distance_line_plots/{model_name}/{actor_name}_{seq_name}_{resolution}x_{frame_id}", f"render_hxw_{render_width}x{render_height}")
    print(f"Plots will be saved to: {base_save_dir}")
    
    # Path to the merging evaluation directory
    merging_evaluation_dir = os.path.join(WORK_DIR, f"results/{model_name}/{actor_name}/{seq_name}/resolution_{resolution}/{frame_id}/viewer_poses/render_hxw_{render_width}x{render_height}/static_merging/merging_evaluation")
    
    # Discover available merging methods
    available_methods = discover_merging_methods(merging_evaluation_dir)
    
    if not available_methods:
        print("No merging method data found!")
        exit(1)
    
    print(f"Found merging methods: {available_methods}")
    
    # Generate plots for each merging method
    print(f"\n{'='*80}")
    print("GENERATING MERGING QUALITY PLOTS")
    print(f"{'='*80}")
    
    for method_name in available_methods:
        print(f"\n{'='*80}")
        print(f"Processing {method_name.replace('_', ' ').title()}")
        print(f"{'='*80}")
        
        generate_all_plots_for_method(merging_evaluation_dir, base_save_dir, method_name)
        
        print(f"Completed plots for {method_name}")
    
    print(f"\n{'='*80}")
    print("ALL MERGING QUALITY VS DISTANCE PLOTTING COMPLETED!")
    print(f"Plots saved under: {base_save_dir}")
    print(f"{'='*80}")
