import os
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lod.voxelization.voxelization import Voxelization, read_gs_frame
from lod.voxelization.voxel_methods import AggregationMethod
from lod.camera_bounds_from_colmap import CameraBounds

from examples.config import Config, load_config_from_toml, merge_config
from scripts.utils import set_result_dir, set_result_dir_voxelize

if __name__ == "__main__":
    # build default config
    default_cfg = Config(strategy=None)
    
    # read the template of yaml from file
    template_path = "./configs/actorshq_voxelize.toml"
    cfg = load_config_from_toml(template_path)
    cfg = merge_config(default_cfg, cfg)
    trained_iter = 30000

    start_frame_id = 0
    end_frame_id = 0
    
    # Test some effective methods
    # aggregate_methods = [
    #     AggregationMethod.scales_voxel_width
    # ]
    aggregate_methods = [
                            AggregationMethod.sum,  # Sum of gaussians
                            AggregationMethod.dominant,  # Dominant gaussian
                            AggregationMethod.mean,  # Mean of gaussians
                            AggregationMethod.kl_moment,  # KL(p||q) first‑/second‑moment match
                            AggregationMethod.kl_moment_hybrid_mean_moment,  # KL(p||q) for 1st moment
                            AggregationMethod.h3dgs,  # H3DGS aggregation method
                            AggregationMethod.h3dgs_hybrid_mean_moment,  # H3DGS for 1st moment
                            # AggregationMethod.w2  # Wasserstein-2 distance aggregation method (Not working)
                        ]
    
    # Test all other methods
    # aggregate_methods = [
    #     AggregationMethod.geometric,
    #     AggregationMethod.adaptive,
    #     AggregationMethod.covariance_weighted,  # Mathematical covariance combination
    #     AggregationMethod.gaussian_mixture,  # Proper Gaussian mixture reduction
    #     AggregationMethod.opacity_preserved,  # Focus on opacity preservation
    #     AggregationMethod.hybrid_dominant,  # Hybrid approach
    #     AggregationMethod.volume_weighted  # Volume-based weighting
    # ]
    
    voxel_sizes = [0.001, 0.0025, 0.005, 0.01]
    for frame_id in range(start_frame_id, end_frame_id + 1):
        cfg.data_dir = os.path.join(cfg.actorshq_data_dir, f"{frame_id}", f"resolution_{cfg.resolution}")
        exp_name = f"actorshq_l1_{1.0 - cfg.ssim_lambda}_ssim_{cfg.ssim_lambda}"
        if cfg.masked_l1_loss:
            exp_name += f"_ml1_{cfg.masked_l1_lambda}"
        if cfg.masked_ssim_loss:
            exp_name += f"_mssim_{cfg.masked_ssim_lambda}"
        if cfg.alpha_loss:
            exp_name += f"_alpha_{cfg.alpha_lambda}"
        if cfg.scale_var_loss:
            exp_name += f"_svar_{cfg.scale_var_lambda}"
        if cfg.random_bkgd:
            exp_name += "_rbkgd"
        cfg.exp_name = exp_name
        cfg.scene_id = frame_id
        
        set_result_dir(cfg, exp_name=exp_name)
        trained_model_path = cfg.result_dir
        
        print(f"Reading gsplat frame at highest resolution for frame {frame_id} ...")
        trained_model = read_gs_frame(trained_model_path, trained_iter)
        
        for voxel_size in voxel_sizes:
            print(f"\nVoxelizing at size: {voxel_size}")
            bounds_calculator = CameraBounds(cfg.data_dir, factor=cfg.data_factor, normalize=False)
            bounds = bounds_calculator.get_scene_center_bounds(radius_multiplier=1.5, padding=0.5)
            voxelizer = Voxelization(voxel_size, bounds=bounds)
            voxelizer.read_gs(trained_model)

            for aggregate_method in aggregate_methods:
                print(f"\nVoxelizing with aggregate method: {aggregate_method}")
                voxelized_model = voxelizer.voxelize(aggregate_method)
            
                # Save voxelized model
                set_result_dir_voxelize(cfg, exp_name=exp_name, voxel_size=voxel_size, aggregate_method=aggregate_method)
                save_model_path = os.path.join(cfg.result_dir, "ckpts/ckpt_0_rank0.pt")
                voxelizer.save_voxelized_gs(save_model_path)
                
                # Save voxelization statistics
                voxelizer.voxel_stats(save_dir=cfg.result_dir)
                
                # Print statistics
                print(f"Original gaussians: {len(trained_model['splats']['means'])}")
                print(f"Voxelized gaussians: {len(voxelized_model['splats']['means'])}")









