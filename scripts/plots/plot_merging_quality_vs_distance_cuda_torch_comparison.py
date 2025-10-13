#!/usr/bin/env python3
"""
Plot comparison between CUDA and Torch implementations for Gaussian merging methods.
Generates comparative plots showing performance differences across different thresholds.
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

def discover_merging_methods(merging_evaluation_dir, implementation):
    """
    Discover available merging methods from the directory structure for a specific implementation.
    
    Args:
        merging_evaluation_dir: Path to the static_merging directory
        implementation: "cuda" or "torch"
        
    Returns:
        List of method names
    """
    impl_dir = os.path.join(merging_evaluation_dir, implementation, "merging_evaluation")
    if not os.path.exists(impl_dir):
        return []
    
    methods = []
    for item in os.listdir(impl_dir):
        item_path = os.path.join(impl_dir, item)
        if os.path.isdir(item_path) and item.startswith('merge_'):
            methods.append(item)
    
    return sorted(methods)


def discover_pixel_thresholds_for_method(merging_evaluation_dir, implementation, method_name):
    """
    Discover available pixel thresholds for a specific merging method and implementation.
    
    Args:
        merging_evaluation_dir: Path to the static_merging directory
        implementation: "cuda" or "torch"
        method_name: Name of the merging method
        
    Returns:
        List of all pixel thresholds found across all parameter configurations
    """
    method_dir = os.path.join(merging_evaluation_dir, implementation, "merging_evaluation", method_name)
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


def load_implementation_data(merging_evaluation_dir, implementation, method_name, metric_name):
    """
    Load metric data for a specific implementation and merging method.
    
    Args:
        merging_evaluation_dir: Path to static_merging directory
        implementation: "cuda" or "torch"
        method_name: Method name (e.g., "merge_center_in_pixel_weighted_px")
        metric_name: Name of the metric (e.g., "psnr", "masked_psnr", "ssim", etc.)
    
    Returns:
        dict: {pixel_threshold: {'distances': [...], 'values': [...]}}
    """
    pixel_thresholds = discover_pixel_thresholds_for_method(merging_evaluation_dir, implementation, method_name)
    if not pixel_thresholds:
        return {}
    
    impl_data = {}
    
    # Find the parameter configuration directory that contains our data
    method_dir = os.path.join(merging_evaluation_dir, implementation, "merging_evaluation", method_name)
    param_dirs = sorted(os.listdir(method_dir))
    
    for pixel_threshold in pixel_thresholds:
        # Find which parameter directory contains this pixel threshold
        found_data = False
        for param_dir in param_dirs:
            param_path = os.path.join(method_dir, param_dir)
            if os.path.isdir(param_path):
                json_file = os.path.join(param_path, f"detailed_results_{pixel_threshold}.json")
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    if "results" not in data:
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
                    
                    impl_data[pixel_threshold] = {
                        'distances': distances_sorted,
                        'values': values_sorted
                    }
                    
                    found_data = True
                    break
        
        if not found_data:
            print(f"Warning: Could not find data for {implementation} pixel threshold {pixel_threshold}")
    
    return impl_data


def plot_cuda_vs_torch_comparison(merging_evaluation_dir, save_dir, method_name, metric_name, metric_label, y_limit=None):
    """Create comparison plot between CUDA and Torch implementations."""
    
    # Load data for both implementations
    cuda_data = load_implementation_data(merging_evaluation_dir, "cuda", method_name, metric_name)
    torch_data = load_implementation_data(merging_evaluation_dir, "torch", method_name, metric_name)
    
    if not cuda_data and not torch_data:
        print(f"No data found for {metric_name} in {method_name}")
        return
    
    # Get all unique pixel thresholds from both implementations
    all_thresholds = sorted(list(set(list(cuda_data.keys()) + list(torch_data.keys()))))
    
    if not all_thresholds:
        print(f"No pixel thresholds found for {metric_name}")
        return
    
    print(f"Found pixel thresholds: {all_thresholds}")
    
    # Assign colors based on pixel thresholds
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_thresholds)))
    color_map = {threshold: colors[i] for i, threshold in enumerate(all_thresholds)}
    
    # Markers for different thresholds
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    marker_map = {threshold: markers[i % len(markers)] for i, threshold in enumerate(all_thresholds)}
    
    plt.figure(figsize=(14, 8))
    
    # Plot CUDA data (solid lines)
    for pixel_threshold in all_thresholds:
        if pixel_threshold in cuda_data:
            data = cuda_data[pixel_threshold]
            label = f'CUDA - Threshold {pixel_threshold}'
            plt.plot(data['distances'], data['values'], 
                    color=color_map[pixel_threshold], 
                    marker=marker_map[pixel_threshold],
                    linestyle='-',  # solid line for CUDA
                    linewidth=2.5, 
                    markersize=6, 
                    alpha=0.8,
                    label=label)
    
    # Plot Torch data (dashed lines)
    for pixel_threshold in all_thresholds:
        if pixel_threshold in torch_data:
            data = torch_data[pixel_threshold]
            label = f'Torch - Threshold {pixel_threshold}'
            plt.plot(data['distances'], data['values'], 
                    color=color_map[pixel_threshold], 
                    marker=marker_map[pixel_threshold],
                    linestyle='--',  # dashed line for Torch
                    linewidth=2.5, 
                    markersize=6, 
                    alpha=0.8,
                    label=label)
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.title(f'{metric_label} vs Camera Distance - CUDA vs Torch Comparison\n{method_name.replace("_", " ").title()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if y_limit:
        plt.ylim(*y_limit)
    
    # Add legend with two columns for better organization
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"{metric_name}_cuda_vs_torch.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"{metric_label} comparison plot saved to: {plot_path}")
    plt.close()


def plot_combined_cuda_torch_metrics(merging_evaluation_dir, save_dir, method_name, metric_pairs):
    """Create combined plots with CUDA vs Torch comparison for multiple metrics."""
    
    for plot_name, (metric1, label1, metric2, label2) in metric_pairs.items():
        cuda_data1 = load_implementation_data(merging_evaluation_dir, "cuda", method_name, metric1)
        torch_data1 = load_implementation_data(merging_evaluation_dir, "torch", method_name, metric1)
        cuda_data2 = load_implementation_data(merging_evaluation_dir, "cuda", method_name, metric2)
        torch_data2 = load_implementation_data(merging_evaluation_dir, "torch", method_name, metric2)
        
        if (not cuda_data1 and not torch_data1) or (not cuda_data2 and not torch_data2):
            print(f"Missing data for combined plot {plot_name}")
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
        
        # Get all thresholds for metric 1
        all_thresholds1 = sorted(list(set(list(cuda_data1.keys()) + list(torch_data1.keys()))))
        colors1 = plt.cm.viridis(np.linspace(0, 1, len(all_thresholds1)))
        color_map1 = {threshold: colors1[i] for i, threshold in enumerate(all_thresholds1)}
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        marker_map1 = {threshold: markers[i % len(markers)] for i, threshold in enumerate(all_thresholds1)}
        
        # Left subplot: First metric
        for pixel_threshold in all_thresholds1:
            if pixel_threshold in cuda_data1:
                data = cuda_data1[pixel_threshold]
                label = f'CUDA - {pixel_threshold}'
                ax1.plot(data['distances'], data['values'], 
                        color=color_map1[pixel_threshold], marker=marker_map1[pixel_threshold],
                        linestyle='-', linewidth=2.5, markersize=6, alpha=0.8, label=label)
            
            if pixel_threshold in torch_data1:
                data = torch_data1[pixel_threshold]
                label = f'Torch - {pixel_threshold}'
                ax1.plot(data['distances'], data['values'], 
                        color=color_map1[pixel_threshold], marker=marker_map1[pixel_threshold],
                        linestyle='--', linewidth=2.5, markersize=6, alpha=0.8, label=label)
        
        ax1.set_xlabel('Distance to Bounding Box Center (m)', fontsize=12)
        ax1.set_ylabel(label1, fontsize=12)
        ax1.set_title(f'{label1} vs Camera Distance', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        if metric1 in ["psnr", "masked_psnr"]:
            ax1.set_ylim(0.0, 80.0)
        elif metric1 in ["ssim", "masked_ssim"]:
            ax1.set_ylim(0.0, 1.0)
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Get all thresholds for metric 2
        all_thresholds2 = sorted(list(set(list(cuda_data2.keys()) + list(torch_data2.keys()))))
        colors2 = plt.cm.viridis(np.linspace(0, 1, len(all_thresholds2)))
        color_map2 = {threshold: colors2[i] for i, threshold in enumerate(all_thresholds2)}
        marker_map2 = {threshold: markers[i % len(markers)] for i, threshold in enumerate(all_thresholds2)}
        
        # Right subplot: Second metric
        for pixel_threshold in all_thresholds2:
            if pixel_threshold in cuda_data2:
                data = cuda_data2[pixel_threshold]
                label = f'CUDA - {pixel_threshold}'
                ax2.plot(data['distances'], data['values'], 
                        color=color_map2[pixel_threshold], marker=marker_map2[pixel_threshold],
                        linestyle='-', linewidth=2.5, markersize=6, alpha=0.8, label=label)
            
            if pixel_threshold in torch_data2:
                data = torch_data2[pixel_threshold]
                label = f'Torch - {pixel_threshold}'
                ax2.plot(data['distances'], data['values'], 
                        color=color_map2[pixel_threshold], marker=marker_map2[pixel_threshold],
                        linestyle='--', linewidth=2.5, markersize=6, alpha=0.8, label=label)
        
        ax2.set_xlabel('Distance to Bounding Box Center (m)', fontsize=12)
        ax2.set_ylabel(label2, fontsize=12)
        ax2.set_title(f'{label2} vs Camera Distance', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        if metric2 in ["psnr", "masked_psnr"]:
            ax2.set_ylim(0.0, 80.0)
        elif metric2 in ["ssim", "masked_ssim"]:
            ax2.set_ylim(0.0, 1.0)
        
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, f"{plot_name}_cuda_vs_torch_combined.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Combined {plot_name} comparison plot saved to: {plot_path}")
        plt.close()


def plot_gaussians_cuda_vs_torch(merging_evaluation_dir, save_dir, method_name):
    """Plot Number of Gaussians comparison between CUDA and Torch."""
    
    cuda_data = load_implementation_data(merging_evaluation_dir, "cuda", method_name, "processed_count")
    torch_data = load_implementation_data(merging_evaluation_dir, "torch", method_name, "processed_count")
    
    if not cuda_data and not torch_data:
        print(f"No data found for Gaussians count")
        return
    
    all_thresholds = sorted(list(set(list(cuda_data.keys()) + list(torch_data.keys()))))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_thresholds)))
    color_map = {threshold: colors[i] for i, threshold in enumerate(all_thresholds)}
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    marker_map = {threshold: markers[i % len(markers)] for i, threshold in enumerate(all_thresholds)}
    
    plt.figure(figsize=(14, 8))
    
    # Get total gaussians count
    total_gaussians = None
    cuda_original = load_implementation_data(merging_evaluation_dir, "cuda", method_name, "original_count")
    if cuda_original:
        for _, o_data in cuda_original.items():
            if o_data['values']:
                total_gaussians = o_data['values'][0]
                break
    
    # Plot CUDA data
    for pixel_threshold in all_thresholds:
        if pixel_threshold in cuda_data:
            data = cuda_data[pixel_threshold]
            label = f'CUDA - Threshold {pixel_threshold}'
            plt.plot(data['distances'], data['values'], 
                    color=color_map[pixel_threshold], marker=marker_map[pixel_threshold],
                    linestyle='-', linewidth=2.5, markersize=6, alpha=0.8, label=label)
    
    # Plot Torch data
    for pixel_threshold in all_thresholds:
        if pixel_threshold in torch_data:
            data = torch_data[pixel_threshold]
            label = f'Torch - Threshold {pixel_threshold}'
            plt.plot(data['distances'], data['values'], 
                    color=color_map[pixel_threshold], marker=marker_map[pixel_threshold],
                    linestyle='--', linewidth=2.5, markersize=6, alpha=0.8, label=label)
    
    # Add horizontal line for total gaussians
    if total_gaussians is not None:
        plt.axhline(y=total_gaussians, color='red', linestyle=':', linewidth=2, 
                   alpha=0.8, label=f'Total Gaussians (no merging): {total_gaussians:,}')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel('Number of Gaussians after Merging', fontsize=12)
    plt.title(f'Gaussians after Merging vs Camera Distance - CUDA vs Torch\n{method_name.replace("_", " ").title()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.ylim(bottom=0)
    
    plot_path = os.path.join(save_dir, "gaussians_cuda_vs_torch.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gaussians comparison plot saved to: {plot_path}")
    plt.close()


def plot_reduction_ratio_cuda_vs_torch(merging_evaluation_dir, save_dir, method_name):
    """Plot Merging Reduction Ratio comparison between CUDA and Torch."""
    
    cuda_data = load_implementation_data(merging_evaluation_dir, "cuda", method_name, "processing_ratio")
    torch_data = load_implementation_data(merging_evaluation_dir, "torch", method_name, "processing_ratio")
    
    if not cuda_data and not torch_data:
        print(f"No data found for reduction ratio")
        return
    
    all_thresholds = sorted(list(set(list(cuda_data.keys()) + list(torch_data.keys()))))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_thresholds)))
    color_map = {threshold: colors[i] for i, threshold in enumerate(all_thresholds)}
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    marker_map = {threshold: markers[i % len(markers)] for i, threshold in enumerate(all_thresholds)}
    
    plt.figure(figsize=(14, 8))
    
    # Plot CUDA data
    for pixel_threshold in all_thresholds:
        if pixel_threshold in cuda_data:
            data = cuda_data[pixel_threshold]
            label = f'CUDA - Threshold {pixel_threshold}'
            plt.plot(data['distances'], data['values'], 
                    color=color_map[pixel_threshold], marker=marker_map[pixel_threshold],
                    linestyle='-', linewidth=2.5, markersize=6, alpha=0.8, label=label)
    
    # Plot Torch data
    for pixel_threshold in all_thresholds:
        if pixel_threshold in torch_data:
            data = torch_data[pixel_threshold]
            label = f'Torch - Threshold {pixel_threshold}'
            plt.plot(data['distances'], data['values'], 
                    color=color_map[pixel_threshold], marker=marker_map[pixel_threshold],
                    linestyle='--', linewidth=2.5, markersize=6, alpha=0.8, label=label)
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel('Merging Reduction Ratio (%)', fontsize=12)
    plt.title(f'Merging Reduction Ratio vs Camera Distance - CUDA vs Torch\n{method_name.replace("_", " ").title()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "reduction_ratio_cuda_vs_torch.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Reduction Ratio comparison plot saved to: {plot_path}")
    plt.close()


def generate_all_cuda_torch_plots(merging_evaluation_dir, base_save_dir, method_name):
    """Generate all comparison plots for CUDA vs Torch."""
    
    # Create method-specific subdirectory
    method_save_dir = os.path.join(base_save_dir, f"{method_name}_cuda_vs_torch")
    os.makedirs(method_save_dir, exist_ok=True)
    
    print(f"\nGenerating CUDA vs Torch comparison plots for {method_name.replace('_', ' ').title()}")
    print(f"Saving to: {method_save_dir}")
    
    # Define all metrics to plot individually
    individual_metrics = [
        ("psnr", "PSNR (dB)", (0.0, 80.0)),
        ("masked_psnr", "Masked PSNR (dB)", (0.0, 80.0)),
        ("ssim", "SSIM", (0.0, 1.0)),
        ("masked_ssim", "Masked SSIM", (0.0, 1.0)),
        ("lpips", "LPIPS", None),
        ("cropped_lpips", "Cropped LPIPS", None)
    ]
    
    # Generate individual metric comparison plots
    for metric_name, metric_label, y_limit in individual_metrics:
        plot_cuda_vs_torch_comparison(merging_evaluation_dir, method_save_dir, method_name, 
                                     metric_name, metric_label, y_limit)
    
    # Define combined plots
    combined_plots = {
        "psnr_comparison": ("psnr", "PSNR (dB)", "masked_psnr", "Masked PSNR (dB)"),
        "ssim_comparison": ("ssim", "SSIM", "masked_ssim", "Masked SSIM"),
        "lpips_comparison": ("lpips", "LPIPS", "cropped_lpips", "Cropped LPIPS")
    }
    
    # Generate combined plots
    plot_combined_cuda_torch_metrics(merging_evaluation_dir, method_save_dir, method_name, combined_plots)
    
    # Generate special analysis plots
    plot_gaussians_cuda_vs_torch(merging_evaluation_dir, method_save_dir, method_name)
    plot_reduction_ratio_cuda_vs_torch(merging_evaluation_dir, method_save_dir, method_name)


if __name__ == "__main__":
    
    model_name = "actorshq_l1_0.5_ssim_0.5_alpha_1.0"
    actor_name = "Actor01"
    seq_name = "Sequence1"
    resolution = 4
    frame_id = 0
    render_height = 1422
    render_width = 2048
    
    print(f"{'='*80}")
    print(f"Plotting CUDA vs Torch Comparison for {model_name}/{actor_name}/{seq_name}/{resolution}x/{frame_id}")
    print(f"Render resolution: {render_width}x{render_height}")
    print(f"{'='*80}")
    
    base_save_dir = os.path.join(WORK_DIR, f"scripts/plots/plot_merging_quality_vs_distance_cuda_torch_comparison/{model_name}/{actor_name}_{seq_name}_{resolution}x_{frame_id}", f"render_hxw_{render_width}x{render_height}")
    print(f"Plots will be saved to: {base_save_dir}")
    
    # Path to the static_merging directory (parent of cuda/ and torch/)
    merging_evaluation_dir = os.path.join(WORK_DIR, f"results/{model_name}/{actor_name}/{seq_name}/resolution_{resolution}/{frame_id}/viewer_poses/render_hxw_{render_width}x{render_height}/static_merging")
    
    # Discover available methods (check both cuda and torch)
    cuda_methods = discover_merging_methods(merging_evaluation_dir, "cuda")
    torch_methods = discover_merging_methods(merging_evaluation_dir, "torch")
    
    # Get methods that exist in at least one implementation
    all_methods = sorted(list(set(cuda_methods + torch_methods)))
    
    if not all_methods:
        print("No merging method data found!")
        exit(1)
    
    print(f"\nFound methods in CUDA: {cuda_methods}")
    print(f"Found methods in Torch: {torch_methods}")
    print(f"Processing methods: {all_methods}")
    
    # Generate plots for each merging method
    print(f"\n{'='*80}")
    print("GENERATING CUDA VS TORCH COMPARISON PLOTS")
    print(f"{'='*80}")
    
    for method_name in all_methods:
        print(f"\n{'='*80}")
        print(f"Processing {method_name.replace('_', ' ').title()}")
        print(f"{'='*80}")
        
        generate_all_cuda_torch_plots(merging_evaluation_dir, base_save_dir, method_name)
        
        print(f"Completed comparison plots for {method_name}")
    
    print(f"\n{'='*80}")
    print("ALL CUDA VS TORCH COMPARISON PLOTTING COMPLETED!")
    print(f"Plots saved under: {base_save_dir}")
    print(f"{'='*80}")

