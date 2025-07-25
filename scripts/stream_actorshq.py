from dataclasses import dataclass
from typing import ClassVar, Optional
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples.simple_streamer import main
from gsplat.distributed import cli
from examples.config import Config, load_config_from_toml, merge_config
from scripts.utils import set_result_dir

# ================= Global Configurations =================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# =========================================================

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

def render_frame(frame_id, iter, config: Config, exp_name, sub_exp_name=None):
    config.data_dir = os.path.join(config.actorshq_data_dir, f"{frame_id}", f"resolution_{config.resolution}")
    config.exp_name = exp_name
    if sub_exp_name is not None:
        config.exp_name = f"{exp_name}_{sub_exp_name}"
    config.run_mode = "render"
    config.scene_id = frame_id
    set_result_dir(config, exp_name=exp_name, sub_exp_name=sub_exp_name)
    ckpt = os.path.join(f"{config.result_dir}/ckpts/ckpt_{iter - 1}_rank0.pt")
    config.ckpt = ckpt
    
    run_experiment(config, dist=False)

if __name__ == '__main__':  
    # build default config
    default_cfg = Config()
    
    # read the template of yaml from file
    template_path = "./configs/actorshq_stream.toml"
    cfg = load_config_from_toml(template_path)
    cfg = merge_config(default_cfg, cfg)

    print(cfg)

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
    frame_id = 0
    
    print(f"\nRendering frame {frame_id}")
    render_frame(frame_id, iter, cfg, exp_name)

'''
CUDA_VISIBLE_DEVICES=0 python scripts/stream_actorshq.py
'''