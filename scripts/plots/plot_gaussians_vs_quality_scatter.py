#!/usr/bin/env python3
"""
Create scatter plots for culling evaluation analysis with flipped axes:
- X-axis: Quality metric value
- Y-axis: Number of Gaussians after culling
- Color: Distance to bounding box center
- Size: Pixel threshold (with options for uniform size)
This provides insights into computational cost vs quality trade-offs.
Modifications:
1. Remove pixel threshold size = 0.0 for distance_only and both_culling
2. Reduce sphere sizes
3. Add uniform size option
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

WORK_DIR = "/main/rajrup/Dropbox/Project/GsplatStream/gsplat"

def discover_pixel_sizes(culling_evaluation_dir):
    """
    Discover available pixel sizes from the detailed_results_px_*.json files.
    Reused from plot_psnr_vs_distance.py
    """
    pattern = os.path.join(culling_evaluation_dir, "detailed_results_px_*.json")
    files = glob.glob(pattern)
    
    pixel_sizes = []
    for file_path in files:
        match = re.search(r'detailed_results_px_([0-9]+\.?[0-9]*)\.json', os.path.basename(file_path))
        if match:
            pixel_sizes.append(float(match.group(1)))
    
    return sorted(pixel_sizes)


def load_gaussians_vs_quality_data(culling_evaluation_dir, culling_method, quality_metric):
    """
    Load all data for gaussians vs quality scatter plots.
    
    Args:
        culling_evaluation_dir: Path to evaluation data
        culling_method: "frustum_only", "distance_only", or "both_culling" 
        quality_metric: Name of quality metric (e.g., "psnr", "masked_psnr", "ssim", etc.)
    
    Returns:
        dict with lists: {
            'quality_values': [qual1, qual2, ...], 
            'num_gaussians': [gauss1, gauss2, ...],
            'distances': [dist1, dist2, ...],
            'pixel_thresholds': [px1, px2, ...],
            'culling_ratios': [ratio1, ratio2, ...]
        }
    """
    pixel_sizes = discover_pixel_sizes(culling_evaluation_dir)
    if not pixel_sizes:
        return {}
    
    all_data = {
        'quality_values': [],
        'num_gaussians': [],
        'distances': [],
        'pixel_thresholds': [], 
        'culling_ratios': []
    }
    
    for pixel_size in pixel_sizes:
        json_file = os.path.join(culling_evaluation_dir, f"detailed_results_px_{pixel_size}.json")
        if not os.path.exists(json_file):
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Construct the full method name based on culling method
        if culling_method == "frustum_only":
            full_method_name = "frustum_only"
            pixel_threshold_for_point = 0.0  # No distance culling
        elif culling_method == "distance_only":
            full_method_name = f"distance_only_px_{pixel_size}"
            pixel_threshold_for_point = pixel_size
        elif culling_method == "both_culling":
            full_method_name = f"both_culling_px_{pixel_size}"
            pixel_threshold_for_point = pixel_size
        else:
            full_method_name = culling_method
            pixel_threshold_for_point = pixel_size
        
        if full_method_name not in data:
            print(f"Method {full_method_name} not found in {json_file}")
            continue
        
        # Extract data for all poses with this pixel size
        for pose_id, metrics in data[full_method_name].items():
            distance = metrics["distance_to_bbox"]
            quality_value = metrics[quality_metric]
            num_gaussians = metrics["culled_count"]
            culling_ratio = metrics["culling_ratio"]
            
            # Skip pixel threshold = 0.0 for distance_only and both_culling methods
            if pixel_threshold_for_point == 0.0 and culling_method in ["distance_only", "both_culling"]:
                continue
            
            # Handle infinity values for PSNR metrics
            if quality_metric in ["psnr", "masked_psnr"] and (quality_value == float('inf') or str(quality_value).lower() == 'infinity'):
                quality_value = 100.0
            
            all_data['quality_values'].append(quality_value)
            all_data['num_gaussians'].append(num_gaussians)
            all_data['distances'].append(distance)
            all_data['pixel_thresholds'].append(pixel_threshold_for_point)
            all_data['culling_ratios'].append(culling_ratio)
    
    return all_data


def plot_gaussians_vs_quality_scatter(culling_evaluation_dir, save_dir, culling_method, quality_metric, quality_label, uniform_size=False):
    """
    Create scatter plot with:
    - X-axis: Quality metric value
    - Y-axis: Number of Gaussians after culling
    - Color: Distance to bounding box center
    - Size: Pixel threshold (variable) or uniform size
    """
    
    scatter_data = load_gaussians_vs_quality_data(culling_evaluation_dir, culling_method, quality_metric)
    if not scatter_data or len(scatter_data['quality_values']) == 0:
        print(f"No data found for {quality_metric} gaussians vs quality scatter plot in {culling_method}")
        return
    
    quality_values = np.array(scatter_data['quality_values'])
    num_gaussians = np.array(scatter_data['num_gaussians'])
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
    
    # Create main scatter plot (flipped axes: quality on x, gaussians on y)
    scatter = ax.scatter(
        quality_values,  # X-axis: Quality
        num_gaussians,   # Y-axis: Gaussians
        c=distances,
        s=normalized_sizes,
        alpha=0.7,
        cmap='viridis',
        norm=distance_norm,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Set labels and title
    ax.set_xlabel(quality_label, fontsize=12)
    ax.set_ylabel('Number of Gaussians after Culling', fontsize=12)
    size_type = "Uniform Size" if uniform_size else "Variable Size"
    ax.set_title(f'Gaussians vs Quality ({size_type}): {quality_label} - {culling_method.replace("_", " ").title()}', fontsize=14)
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
    plot_path = os.path.join(save_dir, f"gaussians_vs_quality_{quality_metric}{suffix}.png")
    if legend is not None:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', bbox_extra_artists=[legend])
    else:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gaussians vs Quality scatter plot for {quality_label} saved to: {plot_path}")
    plt.close()


def plot_culling_ratio_vs_quality(culling_evaluation_dir, save_dir, culling_method, quality_metric, quality_label, uniform_size=False):
    """
    Create scatter plot focused on culling efficiency vs quality:
    - X-axis: Quality metric value 
    - Y-axis: Culling ratio (percentage of gaussians removed)
    - Color: Distance to bounding box center
    - Size: Pixel threshold or uniform
    """
    
    scatter_data = load_gaussians_vs_quality_data(culling_evaluation_dir, culling_method, quality_metric)
    if not scatter_data or len(scatter_data['quality_values']) == 0:
        print(f"No data found for culling ratio vs quality scatter plot in {culling_method}")
        return
    
    quality_values = np.array(scatter_data['quality_values'])
    distances = np.array(scatter_data['distances'])
    pixel_thresholds = np.array(scatter_data['pixel_thresholds'])
    culling_ratios = np.array(scatter_data['culling_ratios'])
    
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
    
    # Create main scatter plot (quality on x, culling ratio on y)
    scatter = ax.scatter(
        quality_values,  # X-axis: Quality
        culling_ratios,  # Y-axis: Culling Ratio
        c=distances,
        s=normalized_sizes,
        alpha=0.7,
        cmap='viridis',
        norm=distance_norm,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Set labels and title
    ax.set_xlabel(quality_label, fontsize=12)
    ax.set_ylabel('Culling Ratio (%)', fontsize=12)
    size_type = "Uniform Size" if uniform_size else "Variable Size"
    ax.set_title(f'Culling Ratio vs Quality ({size_type}): {quality_label} - {culling_method.replace("_", " ").title()}', fontsize=14)
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
    plot_path = os.path.join(save_dir, f"culling_ratio_vs_quality_{quality_metric}{suffix}.png")
    if legend is not None:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', bbox_extra_artists=[legend])
    else:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Culling ratio vs Quality scatter plot saved to: {plot_path}")
    plt.close()


def generate_all_gaussians_vs_quality_plots(culling_evaluation_dir, base_save_dir, culling_method):
    """Generate all gaussians vs quality plots for a specific culling method."""
    
    # Create method-specific subdirectory
    method_save_dir = os.path.join(base_save_dir, f"{culling_method}_gaussians_vs_quality")
    os.makedirs(method_save_dir, exist_ok=True)
    
    print(f"Generating gaussians vs quality plots for {culling_method.replace('_', ' ').title()} in {method_save_dir}")
    
    # Define quality metrics to analyze
    quality_metrics = [
        ("psnr", "PSNR (dB)"),
        ("masked_psnr", "Masked PSNR (dB)"),
        ("ssim", "SSIM"),
        ("masked_ssim", "Masked SSIM"),
        ("lpips", "LPIPS"),
        ("cropped_lpips", "Cropped LPIPS")
    ]
    
    # Generate both variable size and uniform size plots for each quality metric
    for metric_name, metric_label in quality_metrics:
        # Variable size plots
        plot_gaussians_vs_quality_scatter(culling_evaluation_dir, method_save_dir, culling_method, 
                                         metric_name, metric_label, uniform_size=False)
        # Uniform size plots
        plot_gaussians_vs_quality_scatter(culling_evaluation_dir, method_save_dir, culling_method, 
                                         metric_name, metric_label, uniform_size=True)
        
        # Culling ratio vs quality plots for each metric
        plot_culling_ratio_vs_quality(culling_evaluation_dir, method_save_dir, culling_method,
                                     metric_name, metric_label, uniform_size=False)
        plot_culling_ratio_vs_quality(culling_evaluation_dir, method_save_dir, culling_method,
                                     metric_name, metric_label, uniform_size=True)


if __name__ == "__main__":
    
    model_name = "actorshq_l1_0.5_ssim_0.5_alpha_1.0"
    actor_name = "Actor01"
    seq_name = "Sequence1"
    resolution = 4
    frame_id = 0
    # render_height = 1422   # 1422
    # render_width = 2048    # 2048
    # render_height = 711   # 1/2 of 1422
    # render_width = 1024   # 1/2 of 2048
    render_height = 356   # 1/4 of 1422
    render_width = 512   # 1/4 of 2048

    print(f"{'='*80}")
    print(f"Plotting gaussians vs quality scatter plots for {model_name}/{actor_name}/{seq_name}/{resolution}x/{frame_id} with render resolution {render_width}x{render_height}")
    print(f"{'='*80}")

    base_save_dir = os.path.join(WORK_DIR, f"scripts/plots/culling_gaussians_vs_quality_scatter_plots/{model_name}/{actor_name}_{seq_name}_{resolution}x_{frame_id}", f"render_hxw_{render_width}x{render_height}")
    print(f"Gaussians vs Quality plots will be saved to: {base_save_dir}")
    
    # Path to the culling evaluation directory
    culling_evaluation_dir = os.path.join(WORK_DIR, f"results/{model_name}/{actor_name}/{seq_name}/resolution_{resolution}/{frame_id}/viewer_poses/render_hxw_{render_width}x{render_height}/culling_evaluation")
    
    # Define all culling methods to generate plots for
    culling_methods = ["frustum_only", "distance_only", "both_culling"]
    
    # Generate all plots for each culling method
    for culling_method in culling_methods:
        print(f"\n{'='*80}")
        print(f"Processing {culling_method.replace('_', ' ').title()}")
        print(f"{'='*80}")
        
        generate_all_gaussians_vs_quality_plots(culling_evaluation_dir, base_save_dir, culling_method)
        
        print(f"Completed gaussians vs quality plots for {culling_method}")
    
    print(f"\n{'='*80}")
    print("All gaussians vs quality plotting completed!")
    print(f"All plots saved under: {base_save_dir}")
    print(f"{'='*80}")
