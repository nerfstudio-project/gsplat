from dataclasses import dataclass
from typing import ClassVar, Optional
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples.simple_trainer import main, main2
from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from examples.config import Config, load_config_from_toml, merge_config

def extract_path_components(data_dir_path):
    """
    Extract Actor01, Sequence1, and resolution_1 from path like:
    /main/rajrup/Dropbox/Project/GsplatStream/gsplat/data/Actor01/Sequence1/0/resolution_1
    """
    parts = data_dir_path.strip('/').split('/')
    
    # Find the index of 'data' in the path
    try:
        data_index = parts.index('data')
        if data_index + 3 < len(parts):
            actor_name = parts[data_index + 1]  # Actor01
            sequence_name = parts[data_index + 2]  # Sequence1
            resolution_name = parts[data_index + 4]  # resolution_1 (skip frame_id at index 3)
            return actor_name, sequence_name, resolution_name
    except (ValueError, IndexError):
        pass
    
    return None, None, None

def set_result_dir(config: Config, exp_name: Optional[str] = None):
    data_dir = config.data_dir
    scene_id = config.scene_id
    actor, sequence, resolution = extract_path_components(data_dir)
    if exp_name:
        config.result_dir = f"./results/{exp_name}/{actor}/{sequence}/{resolution}/{scene_id}"
    else:
        config.result_dir = f"./results/{actor}/{sequence}/{resolution}/{scene_id}"

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

def train_frame(frame_id, config: Config, exp_name=None):
    config.data_dir = os.path.join(config.data_dir, f"{frame_id}", f"resolution_{config.resolution}")
    config.exp_name = exp_name
    config.scene_id = frame_id
    set_result_dir(config, exp_name)
    config.run_mode = "train"
    config.save_ply = False
    config.save_steps = [7000, 10000, 20000, 30000]
    config.max_steps = 30000
    config.ply_steps = [0, 10000, 20000, 30000]
    # config.eval_steps = [0, 10000, 20000, 30000]
    config.eval_steps = list(sorted(set(range(0, 30001, 10000))))
    
    config.init_type = "sfm"
    config.strategy = DefaultStrategy(verbose=True)
    # config.strategy = StaticPointsStrategy(verbose=False)
    run_experiment(config, dist=False)

def evaluate_frame(frame_id, iter, config: Config, exp_name, sub_exp_name=None):
    config.data_dir = os.path.join(config.data_dir, f"{frame_id}", f"resolution_{config.resolution}")
    config.exp_name = exp_name
    if sub_exp_name is not None:
        config.exp_name = f"{exp_name}_{sub_exp_name}"
    config.run_mode = "eval"
    config.init_type = "sfm"
    config.save_ply = False
    config.scene_id = frame_id
    set_result_dir(config, exp_name=exp_name, sub_exp_name=sub_exp_name)
    ckpt = os.path.join(f"{config.result_dir}/ckpts/ckpt_{iter - 1}_rank0.pt")
    config.ckpt = ckpt
    
    run_experiment(config, dist=False)

@dataclass(frozen=True)
class Method:
    train: ClassVar[str] = "train"
    """ 
    **Training**:  
    Train a gsplat model.
    """
    
    eval: ClassVar[str] = "eval"
    """ 
    **Evaluation**:  
    Evaluate a trained gsplat model.
    """

# ================= Global Configurations =================
method = Method.train
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# =========================================================

if __name__ == '__main__':    
    # build default config
    default_cfg = Config(strategy=DefaultStrategy(verbose=True))
    default_cfg.adjust_steps(default_cfg.steps_scaler)
    
    # read the template of yaml from file
    template_path = "./configs/actorshq.toml"
    cfg = load_config_from_toml(template_path)
    cfg = merge_config(default_cfg, cfg)
    
    if method == Method.eval:
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
        
        cfg.disable_viewer = False
        iter = cfg.max_steps
        start_frame_id = 0
        end_frame_id = 0
        
        for frame_id in range(start_frame_id, end_frame_id + 1):
            print(f"\nEvaluating frame {frame_id}")
            evaluate_frame(frame_id, iter, cfg, exp_name)
    elif method == Method.train:
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
        
        cfg.disable_viewer = True
        start_frame_id = 0
        end_frame_id = 0
        for frame_id in range(start_frame_id, end_frame_id + 1):
            print(f"\nTraining frame {frame_id}")
            train_frame(frame_id, cfg, exp_name)