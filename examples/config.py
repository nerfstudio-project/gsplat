import argparse
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional, List, Union, Literal, Tuple, Any
from typing_extensions import Literal, assert_never
import os
import sys
import toml

from gsplat.strategy import DefaultStrategy, MCMCStrategy

@dataclass
class FreezeParams:
    means: bool = False
    quats: bool = False
    scales: bool = False
    opacities: bool = False
    sh0: bool = False
    shN: bool = False
    colors: bool = False
    features: bool = False

@dataclass
class Config:
    # Experiment name
    exp_name: str = "test"
    
    # Disable viewer
    disable_viewer: bool = False
    # running mode
    run_mode: Literal["train", "eval", "ft", "ft_mv", "ft_of", "resume"] = "train"
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Whether to freeze the splats
    flag_freeze_splats: bool = False
    # Whether to concatenate the splats from previous checkpoint
    flag_concat_splats: bool = False
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: Optional[Literal["interp", "ellipse", "spiral"]] = None
    # Render trajectory FPS
    render_traj_fps: int = 30

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Scene ID/Frame ID
    scene_id: int = 0
    # Resolution of the dataset, only used for actorshq dataset
    resolution: int = 1
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080
    # Background color for the viewer
    viewer_bkgd_color: Literal["", "green", "red", "blue", "white", "black", "random"] = ""

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale based on Avg Distance
    init_scale_avg_dist: bool = True
    # Initial scale of GS
    init_scale: float = 1.0
    
    # Parameters to freeze during training or finetuning
    freeze_splats: bool = False
    freeze_params: Optional[FreezeParams] = None
    
    # Enable Experimental Losses. 
    masked_l1_loss: bool = False    # our experiment
    masked_ssim_loss: bool = False    # our experiment
    alpha_loss: bool = False    # our experiment
    scale_var_loss: bool = False    # our experiment
    # Enable depth loss. (experimental)
    depth_loss: bool = False
    
    # Weights of different losses
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Weight for masked L1 loss
    masked_l1_lambda: float = 1.0
    # Weight for masked SSIM loss
    masked_ssim_lambda: float = 1.0
    # Weight for alpha loss
    alpha_lambda: float = 1.0
    # Weight for scale variance loss
    scale_var_lambda: float = 0.01
    # Weight for depth loss
    depth_lambda: float = 1e-2
    
    # Outlier filtering for point cloud (if using sfm)
    filter_outliers: bool = False
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)

def load_config_from_toml(path: str) -> Config:
    with open(path, 'r') as file:
        config_dict = toml.load(file)
    
    # Handle strategy section separately
    if 'strategy' in config_dict:
        strategy_dict = config_dict.pop('strategy')
        # Check if strategy type is specified, default to DefaultStrategy
        strategy_type = strategy_dict.pop('type', 'default')
        
        try:
            if strategy_type.lower() in ['default', 'defaultstrategy']:
                strategy = DefaultStrategy(**strategy_dict)
            elif strategy_type.lower() in ['mcmc', 'mcmcstrategy']:
                strategy = MCMCStrategy(**strategy_dict)
            else:
                print(f"Unknown strategy type '{strategy_type}', defaulting to DefaultStrategy")
                strategy = DefaultStrategy(**strategy_dict)
        except Exception as e:
            print(f"Error creating strategy with parameters {strategy_dict}: {e}")
            print("Using default strategy parameters")
            strategy = DefaultStrategy()
            
        config_dict['strategy'] = strategy
    
    # Handle freeze_params section separately
    if 'freeze_params' in config_dict:
        freeze_params_dict = config_dict.pop('freeze_params')
        try:
            freeze_params = FreezeParams(**freeze_params_dict)
        except Exception as e:
            print(f"Error creating freeze_params with parameters {freeze_params_dict}: {e}")
            print("Using default freeze_params parameters")
            freeze_params = FreezeParams()
        
        config_dict['freeze_params'] = freeze_params
    
    config = Config(**config_dict)
    return config
        

def merge_config(config_a: Config, config_b: Config) -> Config:
    """
    Only fields in config_b that differ from their default values are used to update config_a.
    """
    def get_default_instance(dataclass_type):
        return dataclass_type() if is_dataclass(dataclass_type) else None

    def should_update(field_name: str, value: Any, default: Any) -> bool:
        return value != default

    def recursive_merge(a: Any, b: Any, a_defaults: Any, b_defaults: Any):
        for field_info in fields(b):
            field_name = field_info.name
            b_value = getattr(b, field_name)
            a_value = getattr(a, field_name)
            default_a = getattr(a_defaults, field_name)
            default_b = getattr(b_defaults, field_name)

            # Determine if the field in B has been explicitly set (i.e., differs from default)
            if should_update(field_name, b_value, default_b):
                if is_dataclass(a_value) and is_dataclass(b_value):
                    # For nested dataclasses, we need to get the correct default instances
                    # of the actual dataclass types, not the parent's defaults
                    a_nested_defaults = get_default_instance(type(a_value))
                    b_nested_defaults = get_default_instance(type(b_value))
                    recursive_merge(a_value, b_value, a_nested_defaults, b_nested_defaults)
                else:
                    setattr(a, field_name, b_value)
    
    config_a_defaults = get_default_instance(Config)
    config_b_defaults = get_default_instance(Config)
    recursive_merge(config_a, config_b, config_a_defaults, config_b_defaults)
    
    return config_a

def main():
    parser = argparse.ArgumentParser(description="Train with TOML configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    default_cfg = Config(strategy=DefaultStrategy(verbose=True))
    
    # Load configuration from TOML file
    cfg = load_config_from_toml(args.config)
    cfg = merge_config(default_cfg, cfg)
    
    # Adjust steps based on steps_scaler
    cfg.adjust_steps(cfg.steps_scaler)
    
    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )
    
    # Print loaded configuration for debugging
    if args.verbose:
        print(f"Loaded configuration from {args.config}")
        print(f"Strategy type: {type(cfg.strategy)}")
        print(f"Strategy parameters:")
        if isinstance(cfg.strategy, DefaultStrategy):
            print(f"  prune_opa: {cfg.strategy.prune_opa}")
            print(f"  grow_grad2d: {cfg.strategy.grow_grad2d}")
            print(f"  prune_scale3d: {cfg.strategy.prune_scale3d}")
            print(f"  refine_start_iter: {cfg.strategy.refine_start_iter}")
            print(f"  refine_stop_iter: {cfg.strategy.refine_stop_iter}")
            print(f"  reset_every: {cfg.strategy.reset_every}")
            print(f"  refine_every: {cfg.strategy.refine_every}")
            print(f"  verbose: {cfg.strategy.verbose}")
        elif isinstance(cfg.strategy, MCMCStrategy):
            print(f"  min_opacity: {cfg.strategy.min_opacity}")
            print(f"  refine_start_iter: {cfg.strategy.refine_start_iter}")
            print(f"  refine_stop_iter: {cfg.strategy.refine_stop_iter}")
            print(f"  refine_every: {cfg.strategy.refine_every}")
        if cfg.freeze_splats and cfg.freeze_params:
            print(f"Freeze params: {cfg.freeze_params}")
        else:
            print("Freeze params: No parameters frozen")
        print(cfg)
    
    # Run the training
    # cli(main2, cfg, verbose=args.verbose)

if __name__ == "__main__":
    main()
    
"""
Script to train using a TOML configuration file.

Usage:
    python examples/config.py --config configs/actorshq.toml --verbose
    python examples/config.py --config configs/voxelize.toml --verbose
"""