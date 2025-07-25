from dataclasses import dataclass
from typing import ClassVar, Optional
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples.simple_trainer import main, main2
from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, StaticPointsStrategy
from lod.voxelization.voxel_methods import AggregationMethod

from examples.config import Config, load_config_from_toml, merge_config, FreezeParams
from scripts.utils import set_result_dir, set_result_dir_voxelize

def run_experiment(config: Config, dist=False):
    print(
            f"------- Running: "
            f"exp_name={config.exp_name}, "
            f"scene_id={config.scene_id}, "
            f"run_mode={config.run_mode}, "
            f"ckpt={config.ckpt} "
            f"---------"
        )
    if dist is True:
        cli(main2, config, verbose=True) # distributed training
    else:
        main2(0, 0, 1, config) # single GPU training

def evaluate_frame(frame_id, iter, config: Config, exp_name, sub_exp_name=None):
    config.data_dir = os.path.join(config.actorshq_data_dir, f"{frame_id}", f"resolution_{config.resolution}")
    config.exp_name = exp_name
    if sub_exp_name is not None:
        config.exp_name = f"{exp_name}_{sub_exp_name}"
    config.run_mode = "eval"
    config.init_type = "sfm"
    config.save_ply = False
    config.freeze_splats = False # Set this to false in eval, otherwise throws error if freeze_splats is true in toml config
    config.scene_id = frame_id
    set_result_dir(config, exp_name=exp_name, sub_exp_name=sub_exp_name)
    ckpt = os.path.join(f"{config.result_dir}/ckpts/ckpt_{iter - 1}_rank0.pt")
    config.ckpt = ckpt
    
    run_experiment(config, dist=False)

def finetune_frame(frame_id, iter, config: Config, exp_name, sub_exp_name=None):
    config.data_dir = os.path.join(config.actorshq_data_dir, f"{frame_id}", f"resolution_{config.resolution}")
    config.exp_name = exp_name
    if sub_exp_name is not None:
        config.exp_name = f"{exp_name}_{sub_exp_name}"
    config.run_mode = "ft"
    config.save_ply = False
    config.scene_id = frame_id
    set_result_dir(config, exp_name=exp_name, sub_exp_name=sub_exp_name)
    ckpt = os.path.join(f"{config.result_dir}/ckpts/ckpt_{iter - 1}_rank0.pt")
    
    config.max_steps = 10000
    config.ply_steps = list(sorted(set(range(0, 30001, 10000)) | {1} ))
    config.eval_steps = list(sorted(set(range(0, 30001, 1000)) | {1} ))
    
    config.init_type = "sfm"
    config.strategy = StaticPointsStrategy(verbose=False)
    # config.strategy = DefaultStrategy(verbose=False)
    # config.strategy.refine_start_iter = 0
    # config.strategy.refine_every = 100
    # config.strategy.reset_every = 1000  # Reset the opacity every 1000 steps
    run_experiment(config, dist=False)

# ================= Global Configurations =================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# =========================================================

if __name__ == '__main__':
    # build default config
    default_cfg = Config(strategy=DefaultStrategy(verbose=True))
    default_cfg.adjust_steps(default_cfg.steps_scaler)
    
    # read the template of yaml from file
    template_path = "./configs/actorshq_voxelize.toml"
    cfg = load_config_from_toml(template_path)
    cfg = merge_config(default_cfg, cfg)
    
    initial_iter = 1
    start_frame_id = 0
    end_frame_id = 0
    
    # Test some effective methods
    aggregate_methods = [
                            AggregationMethod.sum,  # Sum of gaussians
                            AggregationMethod.dominant,  # Dominant gaussian
                            AggregationMethod.mean,  # Mean of gaussians
                            AggregationMethod.kl_moment,  # KL(p||q) first‑/second‑moment match
                            AggregationMethod.kl_moment_hybrid_mean_moment,  # KL(p||q) for 1st moment
                            AggregationMethod.h3dgs,  # H3DGS aggregation method
                            AggregationMethod.h3dgs_hybrid_mean_moment,  # H3DGS for 1st moment
                            # AggregationMethod.w2,  # Wasserstein-2 distance aggregation method
                            # AggregationMethod.scales_voxel_width
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
    
    allowable_voxel_sizes = [0.001, 0.0025, 0.005, 0.01]
    for frame_id in range(start_frame_id, end_frame_id + 1):
        for voxel_size in allowable_voxel_sizes:
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
            
            exp_name = f"{exp_name}_to_voxel_size_{voxel_size}"
            
            if cfg.run_mode == "eval":
                print(f"\nEvaluating frame {frame_id} with voxel size: {voxel_size}")
            elif cfg.run_mode == "ft":
                print(f"\nFinetuning frame {frame_id} with voxel size: {voxel_size}")
            else:
                raise ValueError(f"Invalid run mode: {cfg.run_mode}")
            
            for aggregate_method in aggregate_methods:
                if cfg.run_mode == "eval":
                    print(f"\nEvaluating frame {frame_id} with aggregate method = {aggregate_method}")
                    evaluate_frame(frame_id, initial_iter, cfg, exp_name, sub_exp_name=f"{aggregate_method}")
                elif cfg.run_mode == "ft":
                    print(f"\nFinetuning frame {frame_id} with aggregate method = {aggregate_method}")
                    finetune_frame(frame_id, initial_iter, cfg, exp_name, sub_exp_name=f"{aggregate_method}")

'''
# First voxelize the gsplat model
python scripts/run_actorshq_voxelize.py

# Then finetune/evaluate (change in toml config) the voxelized gsplat model
python scripts/run_actorshq_voxelize_finetune.py

'''