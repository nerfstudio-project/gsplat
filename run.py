from dataclasses import dataclass
from typing import ClassVar
import os

from examples.simple_trainer import main, main2
from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from examples.config import Config, load_config_from_toml, merge_config, set_result_dir

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
        cli(main, config, verbose=True) # distributed training
    else:
        main(0, 0, 1, config) # single GPU training

def train_frame(config: Config):
    config.run_mode = "train"
    config.save_ply = False
    config.save_steps = [7000, 30000]
    config.max_steps = 30000
    config.eval_steps = list(sorted(set(range(0, 30001, 10000))))
    run_experiment(config, dist=False)

def evaluate_frame(config: Config, ckpt):
    config.run_mode = "eval"
    config.init_type = "sfm"
    config.ckpt = ckpt
    config.save_ply = False
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
method = Method.eval
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# =========================================================

if __name__ == '__main__':
    # build default config
    default_cfg = Config(strategy=DefaultStrategy(verbose=True))
    default_cfg.adjust_steps(default_cfg.steps_scaler)
    
    # read the template of yaml from file
    template_path = "./configs/default.toml"
    cfg = load_config_from_toml(template_path)
    cfg = merge_config(default_cfg, cfg)
    
    if method == Method.eval:
        exp_name = f"test"
        cfg.exp_name = exp_name
        set_result_dir(cfg, exp_name=exp_name)
        
        cfg.disable_viewer = False
        iter = cfg.max_steps
        ckpt = [os.path.join(f"{cfg.result_dir}/ckpts/ckpt_{iter - 1}_rank0.pt")]
        print(f"\nEvaluating")
        evaluate_frame(cfg, ckpt)
    elif method == Method.train:
        exp_name = f"test"
        cfg.exp_name = exp_name
        set_result_dir(cfg, exp_name=exp_name)
        
        cfg.disable_viewer = False
        print(f"\nTraining")
        train_frame(cfg)