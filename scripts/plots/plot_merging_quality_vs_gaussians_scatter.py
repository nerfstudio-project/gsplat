#!/usr/bin/env python3
"""
Create scatter plots for merging evaluation analysis with a different perspective:
- X-axis: Number of Gaussians after merging
- Y-axis: Quality metric value
- Color: Distance to bounding box center
- Size: Threshold value (with options for uniform size)

This provides insights into quality vs computational efficiency trade-offs for Gaussian merging.
Modifications from culling version:
1. Remove pixel threshold size = 0.0
2. Reduce sphere sizes
3. Add uniform size option
4. Focus on merging-specific metrics
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

WORK_DIR = "/ssd1/rajrup/Project/gsplat"

def discover_pixel_thresholds_merging(merging_evaluation_dir):
    """
    Discover available pixel thresholds from the detailed_results_px_*.json files.
    Reused from plot_merging_quality_vs_distance_lineplot.py
    """
    pattern = os.path.join(merging_evaluation_dir, "detailed_results_px_*.json")
    files = glob.glob(pattern)
    
    pixel_thresholds = []
    for file_path in files:
        match = re.search(r'detailed_results_px_([0-9]+\.?[0-9]*)\.json', os.path.basename(file_path))
        if match:
            pixel_thresholds.append(float(match.group(1)))
    
    return sorted(pixel_thresholds)


def discover_merging_configurations(merging_evaluation_dir):
    """
    Discover available merging configurations from the JSON files.
    
    Args:
        merging_evaluation_dir: Path to the merging_evaluation directory
        
    Returns:
        Dict mapping pixel thresholds to lists of merging configuration names
    """
    pixel_thresholds = discover_pixel_thresholds_merging(merging_evaluation_dir)
    
    all_configs = {}
    for pixel_threshold in pixel_thresholds:
        json_file = os.path.join(merging_evaluation_dir, f"detailed_results_px_{pixel_threshold}.json")
        if not os.path.exists(json_file):
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract configuration names (exclude any ground truth configs)
        configs = [config_name for config_name in data.keys() 
                  if config_name.startswith('merge_')]
        
        all_configs[pixel_threshold] = configs
    
    return all_configs


def load_merging_quality_vs_gaussians_data(merging_evaluation_dir, merging_method, quality_metric):
    """
    Load all data for merging quality vs gaussians scatter plots.
    
    Args:
        merging_evaluation_dir: Path to evaluation data
        merging_method: Merging method base name (e.g., "merge_knn_k3_maxd0.1_minc2_weighted")
        quality_metric: Name of quality metric (e.g., "psnr", "masked_psnr", "ssim", etc.)
    
    Returns:
        dict with lists: {
            'num_gaussians': [gauss1, gauss2, ...],
            'quality_values': [qual1, qual2, ...], 
            'distances': [dist1, dist2, ...],
            'pixel_thresholds': [px1, px2, ...],
            'merging_ratios': [ratio1, ratio2, ...]
        }
    """
    pixel_thresholds = discover_pixel_thresholds_merging(merging_evaluation_dir)
    if not pixel_thresholds:
        return {}
    
    all_data = {
        'num_gaussians': [],
        'quality_values': [],
        'distances': [],
        'pixel_thresholds': [], 
        'merging_ratios': []
    }
    
    for pixel_threshold in pixel_thresholds:
        json_file = os.path.join(merging_evaluation_dir, f"detailed_results_px_{pixel_threshold}.json")
        if not os.path.exists(json_file):
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Construct the full method name with pixel threshold
        full_method_name = f"{merging_method}_px_{pixel_threshold}"
        pixel_threshold_for_point = pixel_threshold
        
        if full_method_name not in data:
            print(f"Method {full_method_name} not found in {json_file}")
            continue
        
        # Extract data for all poses with this pixel threshold
        for pose_id, metrics in data[full_method_name].items():
            distance = metrics["distance_to_bbox"]
            quality_value = metrics[quality_metric]
            num_gaussians = metrics["processed_count"]
            merging_ratio = metrics["processing_ratio"]
            
            # Skip pixel threshold = 0.0 for merging methods
            if pixel_threshold_for_point == 0.0:
                continue
            
            # Handle infinity values for PSNR metrics
            if quality_metric in ["psnr", "masked_psnr"] and (quality_value == float('inf') or str(quality_value).lower() == 'infinity'):
                quality_value = 100.0
            
            all_data['num_gaussians'].append(num_gaussians)
            all_data['quality_values'].append(quality_value)
            all_data['distances'].append(distance)
            all_data['pixel_thresholds'].append(pixel_threshold_for_point)
            all_data['merging_ratios'].append(merging_ratio)
    
    return all_data


def plot_merging_quality_vs_gaussians_scatter(merging_evaluation_dir, save_dir, merging_method, quality_metric, quality_label, uniform_size=False):
    """
    Create scatter plot with:
    - X-axis: Number of Gaussians after merging
    - Y-axis: Quality metric value
    - Color: Distance to bounding box center
    - Size: Pixel threshold (variable) or uniform size
    """
    
    scatter_data = load_merging_quality_vs_gaussians_data(merging_evaluation_dir, merging_method, quality_metric)
    if not scatter_data or len(scatter_data['num_gaussians']) == 0:
        print(f"No data found for {quality_metric} quality vs gaussians scatter plot in {merging_method}")
        return
    
    num_gaussians = np.array(scatter_data['num_gaussians'])
    quality_values = np.array(scatter_data['quality_values'])
    distances = np.array(scatter_data['distances'])
    pixel_thresholds = np.array(scatter_data['pixel_thresholds'])
    
    # Create figure with extra space for legends
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Normalize distances for color mapping
    distance_norm = Normalize(vmin=np.min(distances), vmax=np.max(distances))
    
    # Get unique pixel thresholds and assign different sizes
    unique_thresholds = sorted(np.unique(pixel_thresholds))
    
    # Create size mapping for pixel thresholds
    if uniform_size:
        # Use uniform size for all points
        normalized_sizes = np.full_like(pixel_thresholds, 50)
        size_range_min, size_range_max = 50, 50
    elif len(unique_thresholds) > 1:
        size_range_min, size_range_max = 20, 120  # Reduced from 30-300 to 20-120
        threshold_min, threshold_max = np.min(pixel_thresholds), np.max(pixel_thresholds)
        
        # Normalize pixel thresholds to size range
        if threshold_max > threshold_min:
            normalized_sizes = (pixel_thresholds - threshold_min) / (threshold_max - threshold_min) * (size_range_max - size_range_min) + size_range_min
        else:
            normalized_sizes = np.full_like(pixel_thresholds, (size_range_min + size_range_max) / 2)
    else:
        # If only one threshold, use medium size
        normalized_sizes = np.full_like(pixel_thresholds, 50)  # Reduced from 100 to 50
        size_range_min, size_range_max = 50, 50
    
    # Create main scatter plot
    scatter = ax.scatter(
        num_gaussians, 
        quality_values,
        c=distances,
        s=normalized_sizes,
        alpha=0.7,
        cmap='viridis',
        norm=distance_norm,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Set labels and title
    ax.set_xlabel('Number of Gaussians after Merging', fontsize=12)
    ax.set_ylabel(quality_label, fontsize=12)
    size_type = "Uniform Size" if uniform_size else "Variable Size"
    ax.set_title(f'Quality vs Gaussians ({size_type}): {quality_label} - {merging_method.replace("_", " ").title()}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create colorbar for distances
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Distance to Bounding Box Center (m) - Color', fontsize=11)
    
    # Create size legend for pixel thresholds
    legend = None
    if uniform_size:
        # For uniform size, omit the pixel threshold legend since all points are the same size
        pass
    elif len(unique_thresholds) > 1:
        # Create legend showing different pixel threshold sizes
        size_legend_elements = []
        
        # Show min, middle, and max thresholds
        legend_thresholds = [unique_thresholds[0]]
        if len(unique_thresholds) > 2:
            legend_thresholds.append(unique_thresholds[len(unique_thresholds)//2])
        legend_thresholds.append(unique_thresholds[-1])
        
        threshold_min, threshold_max = np.min(pixel_thresholds), np.max(pixel_thresholds)
        for threshold in legend_thresholds:
            # Calculate size for this threshold
            if threshold_max > threshold_min:
                size = (threshold - threshold_min) / (threshold_max - threshold_min) * (size_range_max - size_range_min) + size_range_min
            else:
                size = (size_range_min + size_range_max) / 2
                
            size_legend_elements.append(
                plt.scatter([], [], s=size, c='gray', alpha=0.7, edgecolors='black', linewidth=0.5,
                           label=f'Pixel Threshold: {threshold}')
            )
        
        legend = ax.legend(handles=size_legend_elements, title='Pixel Threshold (Size)', 
                          loc='upper left', bbox_to_anchor=(1.15, 0.95),
                          frameon=True, fancybox=True, shadow=True)
    else:
        # Single threshold case
        legend_elements = [plt.scatter([], [], s=normalized_sizes[0], c='gray', alpha=0.7, 
                                     edgecolors='black', linewidth=0.5,
                                     label=f'Pixel Threshold: {unique_thresholds[0]}')]
        legend = ax.legend(handles=legend_elements, title='Pixel Threshold (Size)', 
                          loc='upper left', bbox_to_anchor=(1.15, 0.95),
                          frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to accommodate legends
    plt.subplots_adjust(right=0.75)
    
    # Save plot with bbox_inches='tight' but include legends in the bounding box
    suffix = "_uniform_size" if uniform_size else ""
    plot_path = os.path.join(save_dir, f"quality_vs_gaussians_{quality_metric}{suffix}.png")
    if legend is not None:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', bbox_extra_artists=[legend])
    else:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Quality vs Gaussians scatter plot for {quality_label} saved to: {plot_path}")
    plt.close()


def plot_merging_efficiency_vs_gaussians(merging_evaluation_dir, save_dir, merging_method, uniform_size=False):
    """
    Create scatter plot focused on merging efficiency vs computational cost:
    - X-axis: Number of Gaussians after merging  
    - Y-axis: Merging ratio (percentage of gaussians reduced)
    - Color: Distance to bounding box center
    - Size: Pixel threshold or uniform
    """
    
    scatter_data = load_merging_quality_vs_gaussians_data(merging_evaluation_dir, merging_method, "psnr")
    if not scatter_data or len(scatter_data['num_gaussians']) == 0:
        print(f"No data found for merging efficiency vs gaussians scatter plot in {merging_method}")
        return
    
    num_gaussians = np.array(scatter_data['num_gaussians'])
    distances = np.array(scatter_data['distances'])
    pixel_thresholds = np.array(scatter_data['pixel_thresholds'])
    merging_ratios = np.array(scatter_data['merging_ratios'])
    
    # Create figure with extra space for legends
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Normalize distances for color mapping
    distance_norm = Normalize(vmin=np.min(distances), vmax=np.max(distances))
    
    # Get unique pixel thresholds and assign different sizes
    unique_thresholds = sorted(np.unique(pixel_thresholds))
    
    # Create size mapping for pixel thresholds
    if uniform_size:
        # Use uniform size for all points
        normalized_sizes = np.full_like(pixel_thresholds, 50)
        size_range_min, size_range_max = 50, 50
    elif len(unique_thresholds) > 1:
        size_range_min, size_range_max = 20, 120  # Reduced sizes
        threshold_min, threshold_max = np.min(pixel_thresholds), np.max(pixel_thresholds)
        
        # Normalize pixel thresholds to size range
        if threshold_max > threshold_min:
            normalized_sizes = (pixel_thresholds - threshold_min) / (threshold_max - threshold_min) * (size_range_max - size_range_min) + size_range_min
        else:
            normalized_sizes = np.full_like(pixel_thresholds, (size_range_min + size_range_max) / 2)
    else:
        # If only one threshold, use medium size
        normalized_sizes = np.full_like(pixel_thresholds, 50)
        size_range_min, size_range_max = 50, 50
    
    # Create main scatter plot
    scatter = ax.scatter(
        num_gaussians, 
        merging_ratios,
        c=distances,
        s=normalized_sizes,
        alpha=0.7,
        cmap='viridis',
        norm=distance_norm,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Set labels and title
    ax.set_xlabel('Number of Gaussians after Merging', fontsize=12)
    ax.set_ylabel('Merging Reduction Ratio (%)', fontsize=12)
    size_type = "Uniform Size" if uniform_size else "Variable Size"
    ax.set_title(f'Merging Efficiency vs Computational Cost ({size_type}) - {merging_method.replace("_", " ").title()}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create colorbar for distances
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Distance to Bounding Box Center (m) - Color', fontsize=11)
    
    # Create size legend for pixel thresholds (similar logic as above)
    legend = None
    if uniform_size:
        # For uniform size, omit the pixel threshold legend since all points are the same size
        pass
    elif len(unique_thresholds) > 1:
        # Variable size legend
        size_legend_elements = []
        legend_thresholds = [unique_thresholds[0]]
        if len(unique_thresholds) > 2:
            legend_thresholds.append(unique_thresholds[len(unique_thresholds)//2])
        legend_thresholds.append(unique_thresholds[-1])
        
        threshold_min, threshold_max = np.min(pixel_thresholds), np.max(pixel_thresholds)
        for threshold in legend_thresholds:
            if threshold_max > threshold_min:
                size = (threshold - threshold_min) / (threshold_max - threshold_min) * (size_range_max - size_range_min) + size_range_min
            else:
                size = (size_range_min + size_range_max) / 2
                
            size_legend_elements.append(
                plt.scatter([], [], s=size, c='gray', alpha=0.7, edgecolors='black', linewidth=0.5,
                           label=f'Pixel Threshold: {threshold}')
            )
        
        legend = ax.legend(handles=size_legend_elements, title='Pixel Threshold (Size)', 
                          loc='upper left', bbox_to_anchor=(1.15, 0.95),
                          frameon=True, fancybox=True, shadow=True)
    else:
        legend_elements = [plt.scatter([], [], s=normalized_sizes[0], c='gray', alpha=0.7, 
                                     edgecolors='black', linewidth=0.5,
                                     label=f'Pixel Threshold: {unique_thresholds[0]}')]
        legend = ax.legend(handles=legend_elements, title='Pixel Threshold (Size)', 
                          loc='upper left', bbox_to_anchor=(1.15, 0.95),
                          frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to accommodate legends
    plt.subplots_adjust(right=0.75)
    
    # Save plot with bbox_inches='tight' but include legends in the bounding box
    suffix = "_uniform_size" if uniform_size else ""
    plot_path = os.path.join(save_dir, f"merging_efficiency_vs_gaussians{suffix}.png")
    if legend is not None:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', bbox_extra_artists=[legend])
    else:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Merging efficiency vs Gaussians scatter plot saved to: {plot_path}")
    plt.close()


def generate_all_merging_quality_vs_gaussians_plots(merging_evaluation_dir, base_save_dir, merging_method):
    """Generate all quality vs gaussians plots for a specific merging method."""
    
    # Create method-specific subdirectory
    method_save_dir = os.path.join(base_save_dir, f"{merging_method}_quality_vs_gaussians")
    os.makedirs(method_save_dir, exist_ok=True)
    
    print(f"Generating quality vs gaussians plots for {merging_method.replace('_', ' ').title()} in {method_save_dir}")
    
    # Define quality metrics to analyze
    quality_metrics = [
        ("psnr", "PSNR (dB)"),
        ("masked_psnr", "Masked PSNR (dB)"),
        ("ssim", "SSIM"),
        ("masked_ssim", "Masked SSIM"),
        ("lpips", "LPIPS"),
        ("cropped_lpips", "Cropped LPIPS"),
        ("masked_se", "Masked SE"),
        ("masked_rmse", "Masked RMSE")
    ]
    
    # Generate both variable size and uniform size plots for each quality metric
    for metric_name, metric_label in quality_metrics:
        # Variable size plots
        plot_merging_quality_vs_gaussians_scatter(merging_evaluation_dir, method_save_dir, merging_method, 
                                         metric_name, metric_label, uniform_size=False)
        # Uniform size plots
        plot_merging_quality_vs_gaussians_scatter(merging_evaluation_dir, method_save_dir, merging_method, 
                                         metric_name, metric_label, uniform_size=True)
    
    # Generate merging efficiency vs gaussians plots (both versions)
    plot_merging_efficiency_vs_gaussians(merging_evaluation_dir, method_save_dir, merging_method, uniform_size=False)
    plot_merging_efficiency_vs_gaussians(merging_evaluation_dir, method_save_dir, merging_method, uniform_size=True)


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
    print(f"Plotting merging quality vs gaussians scatter plots for {model_name}/{actor_name}/{seq_name}/{resolution}x/{frame_id} with render resolution {render_width}x{render_height}")
    print(f"{'='*80}")

    base_save_dir = os.path.join(WORK_DIR, f"scripts/plots/static_merging_quality_vs_gaussians_scatter_plots/{model_name}/{actor_name}_{seq_name}_{resolution}x_{frame_id}", f"render_hxw_{render_width}x{render_height}")
    print(f"Quality vs Gaussians plots will be saved to: {base_save_dir}")
    
    # Path to the merging evaluation directory
    merging_evaluation_dir = os.path.join(WORK_DIR, f"results/{model_name}/{actor_name}/{seq_name}/resolution_{resolution}/{frame_id}/viewer_poses/render_hxw_{render_width}x{render_height}/static_merging/merging_evaluation")
    
    # Discover available merging configurations
    all_configs = discover_merging_configurations(merging_evaluation_dir)
    
    if not all_configs:
        print("No merging configuration data found!")
        exit(1)
    
    # Get unique merging methods (remove pixel threshold suffix)
    unique_methods = set()
    for configs in all_configs.values():
        for config in configs:
            # Remove the _px_{threshold} suffix to get the base method name
            base_method = '_'.join(config.split('_')[:-2])  # Remove last two parts: px and threshold
            unique_methods.add(base_method)
    
    print(f"Found merging methods: {list(unique_methods)}")
    
    # Generate plots for each merging method
    print(f"\n{'='*80}")
    print("GENERATING MERGING QUALITY VS GAUSSIANS SCATTER PLOTS")
    print(f"{'='*80}")
    
    for merging_method in sorted(unique_methods):
        print(f"\n{'='*80}")
        print(f"Processing {merging_method.replace('_', ' ').title()}")
        print(f"{'='*80}")
        
        generate_all_merging_quality_vs_gaussians_plots(merging_evaluation_dir, base_save_dir, merging_method)
        
        print(f"Completed quality vs gaussians plots for {merging_method}")
    
    print(f"\n{'='*80}")
    print("ALL MERGING QUALITY VS GAUSSIANS PLOTTING COMPLETED!")
    print(f"Plots saved under: {base_save_dir}")
    print(f"{'='*80}")
