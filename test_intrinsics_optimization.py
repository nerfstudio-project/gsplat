#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# Import gsplat functions
from gsplat import rasterization, rasterization_2dgs

def create_random_scene(n_gaussians, device="cuda:0"):
    """Create a random scene of 3D Gaussians."""
    torch.manual_seed(41)  # For reproducibility
    
    # Random positions in a reasonable range
    means = torch.randn(n_gaussians, 3, device=device) * 3.0
    means[:, 2] += 8.0  # Push gaussians further away from camera (z > 5)
    
    # Random orientations (quaternions)
    quats = F.normalize(torch.randn(n_gaussians, 4, device=device), dim=-1)
    
    # Smaller, more reasonable scales
    scales = torch.exp(torch.randn(n_gaussians, 3, device=device) * 0.3 - 2.0)  # Much smaller scales
    
    # More varied colors (not all purple)
    colors = torch.rand(n_gaussians, 3, device=device)
    
    # Random opacities (not too high)
    opacities = torch.rand(n_gaussians, device=device) * 0.5 + 0.2  # Lower opacity range
    
    return {
        'means': means,
        'quats': quats, 
        'scales': scales,
        'colors': colors,
        'opacities': opacities
    }

def create_camera_setup(device="cuda:0"):
    """Create camera parameters."""
    # Ground truth intrinsics
    fx_true, fy_true = 800.0, 800.0
    cx_true, cy_true = 320.0, 240.0
    
    Ks_true = torch.tensor([[[fx_true, 0.0, cx_true],
                             [0.0, fy_true, cy_true], 
                             [0.0, 0.0, 1.0]]], device=device)
    
    # Initial guess (smaller perturbation)
    fx_init = fx_true + torch.randn(1).item() * 100   # ±100 pixel error
    fy_init = fy_true + torch.randn(1).item() * 100
    cx_init = cx_true + torch.randn(1).item() * 50   # ±50 pixel error  
    cy_init = cy_true + torch.randn(1).item() * 50
    
    Ks_init = torch.tensor([[[fx_init, 0.0, cx_init],
                             [0.0, fy_init, cy_init],
                             [0.0, 0.0, 1.0]]], device=device, requires_grad=True)
    
    # Simple camera pose (identity)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    
    # Image dimensions
    width, height = 640, 480
    
    return {
        'Ks_true': Ks_true,
        'Ks_init': Ks_init,
        'viewmats': viewmats,
        'width': width,
        'height': height
    }

def render_gaussians(scene, camera, Ks, use_2dgs):
    """Render 3D Gaussians to 2D image."""
    # Use the full rasterization pipeline
    if use_2dgs:
        render_colors = rasterization_2dgs(
            means=scene['means'].unsqueeze(0),
            quats=scene['quats'].unsqueeze(0),
            scales=scene['scales'].unsqueeze(0),
            opacities=scene['opacities'].unsqueeze(0),
            colors=scene['colors'].unsqueeze(1).unsqueeze(0),
            viewmats=camera['viewmats'].unsqueeze(0),
            Ks=Ks.unsqueeze(0),
            width=camera['width'],
            height=camera['height'],
            # near_plane=0.01,
            # far_plane=100.0,
            packed=False,  # Use non-packed version which has intrinsics gradients
            backgrounds=torch.zeros(1, 1, 3, device=Ks.device),  # [1, 3] for single camera
            sh_degree=0,
            
        )[0]
        render_colors = render_colors[0]  # Remove batch dimension
    else:
        render_colors = rasterization(
            means=scene['means'],
            quats=scene['quats'],
            scales=scene['scales'],
            opacities=scene['opacities'],
            colors=scene['colors'],
            viewmats=camera['viewmats'],
            Ks=Ks,
            width=camera['width'],
            height=camera['height'],
            packed=False,  # Use non-packed version which has intrinsics gradients
            backgrounds=torch.zeros(1, 3, device=Ks.device),  # [1, 3] for single camera
            camera_model="pinhole"
        )[0]
    return render_colors

def optimize_intrinsics(scene, camera, n_iterations=200, lr=100.0, use_2dgs=None):
    """Optimize camera intrinsics to match ground truth rendering."""

    assert use_2dgs is not None, "use_2dgs must be specified (True or False)"
    
    # Create output directory
    output_dir = Path("intrinsics_optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    # Render ground truth image
    with torch.no_grad():
        gt_image = render_gaussians(scene, camera, camera['Ks_true'], use_2dgs=use_2dgs)
        gt_image = gt_image.squeeze(0)  # Remove batch dimension
    
    # Save ground truth
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(gt_image.cpu().numpy())
    plt.title("Ground Truth")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "00_ground_truth.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Optimization setup
    Ks_opt = camera['Ks_init'].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([Ks_opt], lr=lr)
    # optimizer = torch.optim.SGD([Ks_opt], lr=lr*10, momentum=0.9)
    
    losses = []
    intrinsics_history = []
    
    print("Starting intrinsics optimization...")
    print(f"Ground truth: fx={camera['Ks_true'][0,0,0]:.1f}, fy={camera['Ks_true'][0,1,1]:.1f}, "
          f"cx={camera['Ks_true'][0,0,2]:.1f}, cy={camera['Ks_true'][0,1,2]:.1f}")
    print(f"Initial guess: fx={Ks_opt[0,0,0]:.1f}, fy={Ks_opt[0,1,1]:.1f}, "
          f"cx={Ks_opt[0,0,2]:.1f}, cy={Ks_opt[0,1,2]:.1f}")
    print()
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        # Render with current intrinsics
        pred_image = render_gaussians(scene, camera, Ks_opt, use_2dgs=use_2dgs)
        pred_image = pred_image.squeeze(0)
        
        # Compute loss (MSE between images)
        loss = F.mse_loss(pred_image, gt_image)
        
        # Backward pass
        loss.backward()
        
        # Debug: Check if gradients are being computed
        if iteration == 0:
            print(f"Ks gradients at iteration 0: {Ks_opt.grad}")
            if Ks_opt.grad is None:
                print("WARNING: No gradients computed for Ks!")
            else:
                print(f"Gradient magnitudes: fx={Ks_opt.grad[0,0,0]:.6f}, fy={Ks_opt.grad[0,1,1]:.6f}, "
                      f"cx={Ks_opt.grad[0,0,2]:.6f}, cy={Ks_opt.grad[0,1,2]:.6f}")
        
        optimizer.step()
        
        # Store metrics
        losses.append(loss.item())
        intrinsics_history.append([
            Ks_opt[0,0,0].item(),  # fx
            Ks_opt[0,1,1].item(),  # fy  
            Ks_opt[0,0,2].item(),  # cx
            Ks_opt[0,1,2].item()   # cy
        ])
        
        # Print progress
        if iteration % 20 == 0:
            print(f"Iter {iteration:3d}: Loss={loss.item():.6f}, "
                  f"fx={Ks_opt[0,0,0]:.1f}, fy={Ks_opt[0,1,1]:.1f}, "
                  f"cx={Ks_opt[0,0,2]:.1f}, cy={Ks_opt[0,1,2]:.1f}")
        
        # Save intermediate results
        if iteration % 40 == 0 or iteration == n_iterations - 1:
            plt.figure(figsize=(15, 5))
            
            # Ground truth
            plt.subplot(1, 3, 1)
            plt.imshow(gt_image.cpu().numpy())
            plt.title("Ground Truth")
            plt.axis('off')
            
            # Current prediction
            plt.subplot(1, 3, 2)
            plt.imshow(pred_image.detach().cpu().numpy())
            plt.title(f"Prediction (Iter {iteration})")
            plt.axis('off')
            
            # Difference
            plt.subplot(1, 3, 3)
            diff = torch.abs(pred_image - gt_image).detach().cpu().numpy()
            plt.imshow(diff)
            plt.title(f"Abs Difference (Loss={loss.item():.4f})")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"iter_{iteration:03d}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    # Plot optimization curves
    plt.figure(figsize=(15, 4))
    
    # Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.title('Optimization Progress')
    plt.yscale('log')
    plt.grid(True)
    
    # Intrinsics convergence
    intrinsics_history = np.array(intrinsics_history)
    gt_values = [camera['Ks_true'][0,0,0].item(), camera['Ks_true'][0,1,1].item(),
                 camera['Ks_true'][0,0,2].item(), camera['Ks_true'][0,1,2].item()]
    
    plt.subplot(1, 3, 2)
    labels = ['fx', 'fy', 'cx', 'cy']
    for i, (label, gt_val) in enumerate(zip(labels, gt_values)):
        plt.plot(intrinsics_history[:, i], label=f'{label} (pred)')
        plt.axhline(y=gt_val, color=f'C{i}', linestyle='--', alpha=0.7, label=f'{label} (GT)')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Intrinsics Convergence')
    plt.legend()
    plt.grid(True)
    
    # Final error
    plt.subplot(1, 3, 3)
    final_errors = np.abs(intrinsics_history[-1] - gt_values)
    plt.bar(labels, final_errors)
    plt.ylabel('Absolute Error')
    plt.title('Final Parameter Errors')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "optimization_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Final summary
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print("Ground Truth:")
    print(f"  fx={camera['Ks_true'][0,0,0]:.2f}, fy={camera['Ks_true'][0,1,1]:.2f}")
    print(f"  cx={camera['Ks_true'][0,0,2]:.2f}, cy={camera['Ks_true'][0,1,2]:.2f}")
    print("\nFinal Prediction:")
    print(f"  fx={Ks_opt[0,0,0]:.2f}, fy={Ks_opt[0,1,1]:.2f}")
    print(f"  cx={Ks_opt[0,0,2]:.2f}, cy={Ks_opt[0,1,2]:.2f}")
    print("\nAbsolute Errors:")
    print(f"  fx: {abs(Ks_opt[0,0,0] - camera['Ks_true'][0,0,0]):.2f}")
    print(f"  fy: {abs(Ks_opt[0,1,1] - camera['Ks_true'][0,1,1]):.2f}")
    print(f"  cx: {abs(Ks_opt[0,0,2] - camera['Ks_true'][0,0,2]):.2f}")
    print(f"  cy: {abs(Ks_opt[0,1,2] - camera['Ks_true'][0,1,2]):.2f}")
    print(f"\nFinal Loss: {losses[-1]:.8f}")
    print(f"\nResults saved to: {output_dir.absolute()}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimize camera intrinsics using 3D Gaussian splatting.")
    parser.add_argument('--use_2dgs', action='store_true', help="Use 2DGS rasterization (default is standard rasterization).")
    parser.add_argument('--use_3dgs', action='store_false', dest='use_2dgs', help="Use standard 3D rasterization.")
    args = parser.parse_args()
    use_2dgs = args.use_2dgs
    print(f"Using {'2DGS' if use_2dgs else '3DGS'} rasterization for optimization.")
    """Main function to run the intrinsics optimization test."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create scene and camera
    scene = create_random_scene(n_gaussians=1000, device=device)
    camera = create_camera_setup(device=device)
    
    # Print scene statistics
    print(f"\nScene Statistics:")
    print(f"  Number of Gaussians: {len(scene['means'])}")
    print(f"  Mean positions: {scene['means'].mean(0).cpu().numpy()}")
    print(f"  Position std: {scene['means'].std(0).cpu().numpy()}")
    print(f"  Z range: {scene['means'][:, 2].min().item():.2f} to {scene['means'][:, 2].max().item():.2f}")
    print(f"  Scale range: {scene['scales'].min().item():.4f} to {scene['scales'].max().item():.4f}")
    print(f"  Opacity range: {scene['opacities'].min().item():.3f} to {scene['opacities'].max().item():.3f}")
    
    # Run optimization
    optimize_intrinsics(scene, camera, n_iterations=200, lr=1.0, use_2dgs=use_2dgs)

if __name__ == "__main__":
    main()
