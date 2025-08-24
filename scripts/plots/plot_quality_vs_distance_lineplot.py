#!/usr/bin/env python3
"""
Plot comprehensive evaluation metrics for culling methods across different thresholds.
Supports both pixel-based and area-based culling evaluation:
- Pixel-based: Uses pixel size thresholds from static_dist_culling results
- Area-based: Uses area thresholds from static_dist_area_culling results
Generates plots for PSNR, SSIM, LPIPS, and their masked/cropped variants.
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

def discover_pixel_sizes_threshold(culling_evaluation_dir):
    """
    Discover available pixel sizes from the detailed_results_px_*.json files.
    
    Args:
        culling_evaluation_dir: Path to the culling_evaluation directory
        
    Returns:
        List of pixel sizes (as floats) sorted in ascending order
    """
    pattern = os.path.join(culling_evaluation_dir, "detailed_results_px_*.json")
    files = glob.glob(pattern)
    
    pixel_sizes_threshold = []
    for file_path in files:
        match = re.search(r'detailed_results_px_([0-9]+\.?[0-9]*)\.json', os.path.basename(file_path))
        if match:
            pixel_sizes_threshold.append(float(match.group(1)))
    
    return sorted(pixel_sizes_threshold)


def discover_area_thresholds(culling_evaluation_dir):
    """
    Discover available area thresholds from the detailed_results_area_*.json files.
    
    Args:
        culling_evaluation_dir: Path to the culling_evaluation directory
        
    Returns:
        List of area thresholds (as floats) sorted in ascending order
    """
    pattern = os.path.join(culling_evaluation_dir, "detailed_results_area_*.json")
    files = glob.glob(pattern)
    
    area_thresholds = []
    for file_path in files:
        match = re.search(r'detailed_results_area_([0-9]+\.?[0-9]*)\.json', os.path.basename(file_path))
        if match:
            area_thresholds.append(float(match.group(1)))
    
    return sorted(area_thresholds)


def load_metric_data(culling_evaluation_dir, culling_method, metric_name):
    """
    Load metric data for all pixel sizes and return sorted by distance.
    
    Args:
        culling_evaluation_dir: Path to evaluation data
        culling_method: "frustum_only", "distance_only", or "both_culling"
        metric_name: Name of the metric (e.g., "psnr", "masked_psnr", "ssim", etc.)
    
    Returns:
        dict: {pixel_size_threshold: {'distances': [...], 'values': [...], 'color': ..., 'marker': ...}}
    """
    pixel_sizes_threshold = discover_pixel_sizes_threshold(culling_evaluation_dir)
    if not pixel_sizes_threshold:
        return {}
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(pixel_sizes_threshold)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    all_data = {}
    for i, pixel_size_threshold in enumerate(pixel_sizes_threshold):
        json_file = os.path.join(culling_evaluation_dir, f"detailed_results_px_{pixel_size_threshold}.json")
        if not os.path.exists(json_file):
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Construct the full method name based on culling method
        if culling_method == "frustum_only":
            full_method_name = "frustum_only"
        elif culling_method == "distance_only":
            full_method_name = f"distance_only_px_{pixel_size_threshold}"
        elif culling_method == "both_culling":
            full_method_name = f"both_culling_px_{pixel_size_threshold}"
        else:
            full_method_name = culling_method
        
        if full_method_name not in data:
            print(f"Method {full_method_name} not found in {json_file}")
            continue
            
        distances = []
        values = []
        
        for pose_id, metrics in data[full_method_name].items():
            distance = metrics["distance_to_bbox"]
            value = metrics[metric_name]
            
            # Handle infinity values for PSNR metrics
            if metric_name in ["psnr", "masked_psnr"] and (value == float('inf') or str(value).lower() == 'infinity'):
                value = 100.0
            
            distances.append(distance)
            values.append(value)
        
        # Sort by distance
        sorted_data = sorted(zip(distances, values))
        distances_sorted, values_sorted = zip(*sorted_data)
        
        all_data[pixel_size_threshold] = {
            'distances': distances_sorted,
            'values': values_sorted,
            'color': colors[i],
            'marker': markers[i % len(markers)]
        }
    
    return all_data


def load_area_based_metric_data(culling_evaluation_dir, culling_method, metric_name):
    """
    Load metric data for all area thresholds and return sorted by distance.
    
    Args:
        culling_evaluation_dir: Path to evaluation data
        culling_method: "frustum_only", "distance_only", or "both_culling"
        metric_name: Name of the metric (e.g., "psnr", "masked_psnr", "ssim", etc.)
    
    Returns:
        dict: {area_threshold: {'distances': [...], 'values': [...], 'color': ..., 'marker': ...}}
    """
    area_thresholds = discover_area_thresholds(culling_evaluation_dir)
    if not area_thresholds:
        return {}
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(area_thresholds)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    all_data = {}
    for i, area_threshold in enumerate(area_thresholds):
        json_file = os.path.join(culling_evaluation_dir, f"detailed_results_area_{area_threshold}.json")
        if not os.path.exists(json_file):
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Construct the full method name based on culling method
        if culling_method == "frustum_only":
            full_method_name = "frustum_only"
        elif culling_method == "distance_only":
            full_method_name = f"distance_only_area_{area_threshold}"
        elif culling_method == "both_culling":
            full_method_name = f"both_culling_area_{area_threshold}"
        else:
            full_method_name = culling_method
        
        if full_method_name not in data:
            print(f"Method {full_method_name} not found in {json_file}")
            continue
            
        distances = []
        values = []
        
        for pose_id, metrics in data[full_method_name].items():
            distance = metrics["distance_to_bbox"]
            value = metrics[metric_name]
            
            # Handle infinity values for PSNR metrics
            if metric_name in ["psnr", "masked_psnr"] and (value == float('inf') or str(value).lower() == 'infinity'):
                value = 100.0
            
            distances.append(distance)
            values.append(value)
        
        # Sort by distance
        sorted_data = sorted(zip(distances, values))
        distances_sorted, values_sorted = zip(*sorted_data)
        
        all_data[area_threshold] = {
            'distances': distances_sorted,
            'values': values_sorted,
            'color': colors[i],
            'marker': markers[i % len(markers)]
        }
    
    return all_data


def plot_metric_comparison(culling_evaluation_dir, save_dir, culling_method, metric_name, metric_label, y_limit=None):
    """Create comparison plot for a specific metric across pixel sizes."""
    
    all_data = load_metric_data(culling_evaluation_dir, culling_method, metric_name)
    if not all_data:
        print(f"No data found for {metric_name} in {culling_method}")
        return
    
    plt.figure(figsize=(10, 6))
    for pixel_size_threshold, data in all_data.items():
        plt.plot(data['distances'], data['values'], 
                color=data['color'], marker=data['marker'], 
                linewidth=2, markersize=4, alpha=0.7,
                label=f'Pixel Size {pixel_size_threshold}')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.title(f'{metric_label} vs Camera Distance - {culling_method.replace("_", " ").title()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if y_limit:
        plt.ylim(*y_limit)
    
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"{metric_name}_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"{metric_label} plot saved to: {plot_path}")
    plt.close()  # Close to free memory


def plot_area_based_metric_comparison(culling_evaluation_dir, save_dir, culling_method, metric_name, metric_label, y_limit=None):
    """Create comparison plot for a specific metric across area thresholds."""
    
    all_data = load_area_based_metric_data(culling_evaluation_dir, culling_method, metric_name)
    if not all_data:
        print(f"No area-based data found for {metric_name} in {culling_method}")
        return
    
    plt.figure(figsize=(10, 6))
    for area_threshold, data in all_data.items():
        plt.plot(data['distances'], data['values'], 
                color=data['color'], marker=data['marker'], 
                linewidth=2, markersize=4, alpha=0.7,
                label=f'Area Threshold {area_threshold} px²')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.title(f'Area-Based {metric_label} vs Camera Distance - {culling_method.replace("_", " ").title()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if y_limit:
        plt.ylim(*y_limit)
    
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"area_based_{metric_name}_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Area-based {metric_label} plot saved to: {plot_path}")
    plt.close()  # Close to free memory


def plot_combined_metrics(culling_evaluation_dir, save_dir, culling_method, metric_pairs):
    """Create combined plots with multiple metrics side by side."""
    
    for plot_name, (metric1, label1, metric2, label2) in metric_pairs.items():
        data1 = load_metric_data(culling_evaluation_dir, culling_method, metric1)
        data2 = load_metric_data(culling_evaluation_dir, culling_method, metric2)
        
        if not data1 or not data2:
            print(f"Missing data for combined plot {plot_name}")
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left subplot: First metric
        for pixel_size_threshold, data in data1.items():
            ax1.plot(data['distances'], data['values'], 
                    color=data['color'], marker=data['marker'], 
                    linewidth=2, markersize=4, alpha=0.7,
                    label=f'Pixel Size {pixel_size_threshold}')
        
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
        for pixel_size_threshold, data in data2.items():
            ax2.plot(data['distances'], data['values'], 
                    color=data['color'], marker=data['marker'], 
                    linewidth=2, markersize=4, alpha=0.7,
                    label=f'Pixel Size {pixel_size_threshold}')
        
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


def plot_area_based_combined_metrics(culling_evaluation_dir, save_dir, culling_method, metric_pairs):
    """Create combined plots with multiple metrics side by side for area-based culling."""
    
    for plot_name, (metric1, label1, metric2, label2) in metric_pairs.items():
        data1 = load_area_based_metric_data(culling_evaluation_dir, culling_method, metric1)
        data2 = load_area_based_metric_data(culling_evaluation_dir, culling_method, metric2)
        
        if not data1 or not data2:
            print(f"Missing area-based data for combined plot {plot_name}")
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left subplot: First metric
        for area_threshold, data in data1.items():
            ax1.plot(data['distances'], data['values'], 
                    color=data['color'], marker=data['marker'], 
                    linewidth=2, markersize=4, alpha=0.7,
                    label=f'Area Threshold {area_threshold} px²')
        
        ax1.set_xlabel('Distance to Bounding Box Center (m)', fontsize=12)
        ax1.set_ylabel(label1, fontsize=12)
        ax1.set_title(f'Area-Based {label1} vs Camera Distance', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Set appropriate y-limits based on metric type
        if metric1 in ["psnr", "masked_psnr"]:
            ax1.set_ylim(0.0, 80.0)
        elif metric1 in ["ssim", "masked_ssim"]:
            ax1.set_ylim(0.0, 1.0)
        
        ax1.legend()
        
        # Right subplot: Second metric
        for area_threshold, data in data2.items():
            ax2.plot(data['distances'], data['values'], 
                    color=data['color'], marker=data['marker'], 
                    linewidth=2, markersize=4, alpha=0.7,
                    label=f'Area Threshold {area_threshold} px²')
        
        ax2.set_xlabel('Distance to Bounding Box Center (m)', fontsize=12)
        ax2.set_ylabel(label2, fontsize=12)
        ax2.set_title(f'Area-Based {label2} vs Camera Distance', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Set appropriate y-limits based on metric type
        if metric2 in ["psnr", "masked_psnr"]:
            ax2.set_ylim(0.0, 80.0)
        elif metric2 in ["ssim", "masked_ssim"]:
            ax2.set_ylim(0.0, 1.0)
        
        ax2.legend()
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, f"area_based_{plot_name}_combined.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Area-based combined {plot_name} plot saved to: {plot_path}")
        plt.close()


def plot_gaussians_vs_distance(culling_evaluation_dir, save_dir, culling_method):
    """Plot Number of Gaussians after culling vs Distance."""
    
    pixel_sizes_threshold = discover_pixel_sizes_threshold(culling_evaluation_dir)
    if not pixel_sizes_threshold:
        print("No pixel size data files found!")
        return
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(pixel_sizes_threshold)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Collect data for all pixel sizes
    all_data = {}
    total_gaussians = None  # Should be constant across all poses
    
    for i, pixel_size_threshold in enumerate(pixel_sizes_threshold):
        json_file = os.path.join(culling_evaluation_dir, f"detailed_results_px_{pixel_size_threshold}.json")
        if not os.path.exists(json_file):
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Construct the full method name based on culling method
        if culling_method == "frustum_only":
            full_method_name = "frustum_only"
        elif culling_method == "distance_only":
            full_method_name = f"distance_only_px_{pixel_size_threshold}"
        elif culling_method == "both_culling":
            full_method_name = f"both_culling_px_{pixel_size_threshold}"
        else:
            full_method_name = culling_method
            
        if full_method_name not in data:
            print(f"Method {full_method_name} not found in {json_file}")
            continue
            
        distances = []
        culled_gaussians = []
        wo_culled_gaussians = []
        
        for pose_id, metrics in data[full_method_name].items():
            distance = metrics["distance_to_bbox"]
            num_after_culling = metrics["culled_count"]
            wo_culled_count = metrics["wo_culled_count"]
            
            distances.append(distance)
            culled_gaussians.append(num_after_culling)
            wo_culled_gaussians.append(wo_culled_count)
        
        # Store total gaussians (should be same for all pixel sizes)
        if total_gaussians is None:
            total_gaussians = wo_culled_gaussians[0]  # Take first value as reference
        
        # Sort by distance
        sorted_data = sorted(zip(distances, culled_gaussians))
        distances_sorted, culled_gaussians_sorted = zip(*sorted_data)
        
        all_data[pixel_size_threshold] = {
            'distances': distances_sorted,
            'culled_gaussians': culled_gaussians_sorted,
            'color': colors[i],
            'marker': markers[i % len(markers)]
        }
    
    # Create single plot for Gaussians vs Distance
    plt.figure(figsize=(10, 6))
    
    # Plot Number of Gaussians after culling vs Distance
    for pixel_size_threshold, data in all_data.items():
        plt.plot(data['distances'], data['culled_gaussians'], 
                color=data['color'], marker=data['marker'], 
                linewidth=2, markersize=4, alpha=0.7,
                label=f'Pixel Size {pixel_size_threshold}')
    
    # Add horizontal line for total gaussians (without culling)
    if total_gaussians is not None:
        plt.axhline(y=total_gaussians, color='red', linestyle='--', linewidth=2, 
                   alpha=0.8, label=f'Total Gaussians (no culling): {total_gaussians:,}')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel('Number of Gaussians after Culling', fontsize=12)
    plt.title(f'Gaussians after Culling vs Camera Distance - {culling_method.replace("_", " ").title()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "gaussians_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gaussians vs Distance plot saved to: {plot_path}")
    plt.close()


def plot_area_based_gaussians_vs_distance(culling_evaluation_dir, save_dir, culling_method):
    """Plot Number of Gaussians after culling vs Distance for area-based culling."""
    
    area_thresholds = discover_area_thresholds(culling_evaluation_dir)
    if not area_thresholds:
        print("No area threshold data files found!")
        return
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(area_thresholds)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Collect data for all area thresholds
    all_data = {}
    total_gaussians = None  # Should be constant across all poses
    
    for i, area_threshold in enumerate(area_thresholds):
        json_file = os.path.join(culling_evaluation_dir, f"detailed_results_area_{area_threshold}.json")
        if not os.path.exists(json_file):
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Construct the full method name based on culling method
        if culling_method == "frustum_only":
            full_method_name = "frustum_only"
        elif culling_method == "distance_only":
            full_method_name = f"distance_only_area_{area_threshold}"
        elif culling_method == "both_culling":
            full_method_name = f"both_culling_area_{area_threshold}"
        else:
            full_method_name = culling_method
            
        if full_method_name not in data:
            print(f"Method {full_method_name} not found in {json_file}")
            continue
            
        distances = []
        culled_gaussians = []
        wo_culled_gaussians = []
        
        for pose_id, metrics in data[full_method_name].items():
            distance = metrics["distance_to_bbox"]
            num_after_culling = metrics["culled_count"]
            wo_culled_count = metrics["wo_culled_count"]
            
            distances.append(distance)
            culled_gaussians.append(num_after_culling)
            wo_culled_gaussians.append(wo_culled_count)
        
        # Store total gaussians (should be same for all area thresholds)
        if total_gaussians is None:
            total_gaussians = wo_culled_gaussians[0]  # Take first value as reference
        
        # Sort by distance
        sorted_data = sorted(zip(distances, culled_gaussians))
        distances_sorted, culled_gaussians_sorted = zip(*sorted_data)
        
        all_data[area_threshold] = {
            'distances': distances_sorted,
            'culled_gaussians': culled_gaussians_sorted,
            'color': colors[i],
            'marker': markers[i % len(markers)]
        }
    
    # Create single plot for Gaussians vs Distance
    plt.figure(figsize=(10, 6))
    
    # Plot Number of Gaussians after culling vs Distance
    for area_threshold, data in all_data.items():
        plt.plot(data['distances'], data['culled_gaussians'], 
                color=data['color'], marker=data['marker'], 
                linewidth=2, markersize=4, alpha=0.7,
                label=f'Area Threshold {area_threshold} px²')
    
    # Add horizontal line for total gaussians (without culling)
    if total_gaussians is not None:
        plt.axhline(y=total_gaussians, color='red', linestyle='--', linewidth=2, 
                   alpha=0.8, label=f'Total Gaussians (no culling): {total_gaussians:,}')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel('Number of Gaussians after Culling', fontsize=12)
    plt.title(f'Area-Based Gaussians after Culling vs Camera Distance - {culling_method.replace("_", " ").title()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "area_based_gaussians_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Area-based Gaussians vs Distance plot saved to: {plot_path}")
    plt.close()


def plot_valid_pixels_vs_distance(culling_evaluation_dir, save_dir, culling_method):
    """Plot Number of Valid Pixels vs Distance."""
    
    pixel_sizes_threshold = discover_pixel_sizes_threshold(culling_evaluation_dir)
    if not pixel_sizes_threshold:
        print("No pixel size data files found!")
        return
    
    # Get valid pixels data (should be same across pixel sizes, varies with pose)
    valid_pixels_data = None
    
    for pixel_size_threshold in pixel_sizes_threshold:
        json_file = os.path.join(culling_evaluation_dir, f"detailed_results_px_{pixel_size_threshold}.json")
        if not os.path.exists(json_file):
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Construct the full method name based on culling method
        if culling_method == "frustum_only":
            full_method_name = "frustum_only"
        elif culling_method == "distance_only":
            full_method_name = f"distance_only_px_{pixel_size_threshold}"
        elif culling_method == "both_culling":
            full_method_name = f"both_culling_px_{pixel_size_threshold}"
        else:
            full_method_name = culling_method
            
        if full_method_name not in data:
            print(f"Method {full_method_name} not found in {json_file}")
            continue
            
        distances = []
        valid_pixels = []
        
        for pose_id, metrics in data[full_method_name].items():
            distance = metrics["distance_to_bbox"]
            num_valid_pixels = metrics["num_valid_pixels"]
            
            distances.append(distance)
            valid_pixels.append(num_valid_pixels)
        
        # Sort by distance
        sorted_data = sorted(zip(distances, valid_pixels))
        distances_sorted, valid_pixels_sorted = zip(*sorted_data)
        
        # Store valid pixels data (should be same across pixel sizes)
        if valid_pixels_data is None:
            valid_pixels_data = {
                'distances': distances_sorted,
                'valid_pixels': valid_pixels_sorted
            }
        break  # Only need one pixel size since valid pixels are the same across all
    
    # Create single plot for Valid Pixels vs Distance
    plt.figure(figsize=(10, 6))
    
    # Plot Number of Valid Pixels vs Distance
    if valid_pixels_data is not None:
        plt.plot(valid_pixels_data['distances'], valid_pixels_data['valid_pixels'], 
                color='blue', marker='o', linewidth=2, markersize=4, alpha=0.7,
                label='Valid Pixels (same for all pixel coverage thresholds)')
    
    plt.xlabel('Distance to Bounding Box Center (m)', fontsize=12)
    plt.ylabel('Number of Valid Pixels (log scale)', fontsize=12)
    plt.title(f'Valid Pixels vs Camera Distance - {culling_method.replace("_", " ").title()}', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "valid_pixels_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Valid Pixels vs Distance plot saved to: {plot_path}")
    plt.close()


def generate_all_plots_for_method(culling_evaluation_dir, base_save_dir, culling_method):
    """Generate all plots for a specific culling method."""
    
    # Create method-specific subdirectory
    method_save_dir = os.path.join(base_save_dir, culling_method)
    os.makedirs(method_save_dir, exist_ok=True)
    
    print(f"Generating plots for {culling_method.replace('_', ' ').title()} in {method_save_dir}")
    
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
        plot_metric_comparison(culling_evaluation_dir, method_save_dir, culling_method, 
                             metric_name, metric_label, y_limit)
    
    # Define combined plots (side-by-side comparisons)
    combined_plots = {
        "psnr_comparison": ("psnr", "PSNR (dB)", "masked_psnr", "Masked PSNR (dB)"),
        "ssim_comparison": ("ssim", "SSIM", "masked_ssim", "Masked SSIM"),
        "lpips_comparison": ("lpips", "LPIPS", "cropped_lpips", "Cropped LPIPS")
    }
    
    # Generate combined plots
    plot_combined_metrics(culling_evaluation_dir, method_save_dir, culling_method, combined_plots)
    
    # Generate special analysis plots
    plot_gaussians_vs_distance(culling_evaluation_dir, method_save_dir, culling_method)
    plot_valid_pixels_vs_distance(culling_evaluation_dir, method_save_dir, culling_method)


def generate_all_area_based_plots_for_method(culling_evaluation_dir, base_save_dir, culling_method):
    """Generate all area-based plots for a specific culling method."""
    
    # Create method-specific subdirectory for area-based plots
    method_save_dir = os.path.join(base_save_dir, f"{culling_method}_area_based")
    os.makedirs(method_save_dir, exist_ok=True)
    
    print(f"Generating area-based plots for {culling_method.replace('_', ' ').title()} in {method_save_dir}")
    
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
        plot_area_based_metric_comparison(culling_evaluation_dir, method_save_dir, culling_method, 
                             metric_name, metric_label, y_limit)
    
    # Define combined plots (side-by-side comparisons)
    combined_plots = {
        "psnr_comparison": ("psnr", "PSNR (dB)", "masked_psnr", "Masked PSNR (dB)"),
        "ssim_comparison": ("ssim", "SSIM", "masked_ssim", "Masked SSIM"),
        "lpips_comparison": ("lpips", "LPIPS", "cropped_lpips", "Cropped LPIPS")
    }
    
    # Generate combined plots
    plot_area_based_combined_metrics(culling_evaluation_dir, method_save_dir, culling_method, combined_plots)
    
    # Generate special analysis plots
    plot_area_based_gaussians_vs_distance(culling_evaluation_dir, method_save_dir, culling_method)
    # Note: Valid pixels vs distance is the same for all methods, so we don't need an area-based version


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
    print(f"Plotting Quality vs Distance for {model_name}/{actor_name}/{seq_name}/{resolution}x/{frame_id} with render resolution {render_width}x{render_height}")
    print(f"{'='*80}")

    base_save_dir = os.path.join(WORK_DIR, f"scripts/plots/static_culling_quality_vs_distance_line_plots/{model_name}/{actor_name}_{seq_name}_{resolution}x_{frame_id}", f"render_hxw_{render_width}x{render_height}")
    print(f"Plots will be saved to: {base_save_dir}")
    
    # Paths to the culling evaluation directories
    pixel_culling_evaluation_dir = os.path.join(WORK_DIR, f"results/{model_name}/{actor_name}/{seq_name}/resolution_{resolution}/{frame_id}/viewer_poses/render_hxw_{render_width}x{render_height}/static_dist_culling/culling_evaluation")
    area_culling_evaluation_dir = os.path.join(WORK_DIR, f"results/{model_name}/{actor_name}/{seq_name}/resolution_{resolution}/{frame_id}/viewer_poses/render_hxw_{render_width}x{render_height}/static_dist_area_culling/culling_evaluation")
    
    # Define all culling methods to generate plots for
    culling_methods = ["frustum_only", "distance_only", "both_culling"]
    
    # Generate pixel-based plots for each culling method
    print(f"\n{'='*80}")
    print("GENERATING PIXEL-BASED CULLING PLOTS")
    print(f"{'='*80}")
    
    for culling_method in culling_methods:
        print(f"\n{'='*80}")
        print(f"Processing Pixel-Based {culling_method.replace('_', ' ').title()}")
        print(f"{'='*80}")
        
        generate_all_plots_for_method(pixel_culling_evaluation_dir, base_save_dir, culling_method)
        
        print(f"Completed pixel-based plots for {culling_method}")
    
    # Generate area-based plots for each culling method
    print(f"\n{'='*80}")
    print("GENERATING AREA-BASED CULLING PLOTS")
    print(f"{'='*80}")
    
    for culling_method in culling_methods:
        print(f"\n{'='*80}")
        print(f"Processing Area-Based {culling_method.replace('_', ' ').title()}")
        print(f"{'='*80}")
        
        generate_all_area_based_plots_for_method(area_culling_evaluation_dir, base_save_dir, culling_method)
        
        print(f"Completed area-based plots for {culling_method}")
    
    print(f"\n{'='*80}")
    print("ALL QUALITY VS DISTANCE PLOTTING COMPLETED!")
    print(f"Pixel-based plots saved under: {base_save_dir}")
    print(f"Area-based plots saved under: {base_save_dir}")
    print(f"{'='*80}")