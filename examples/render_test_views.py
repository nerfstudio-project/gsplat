import argparse
import os
from typing import Dict, Literal, Optional, Tuple

import torch
import yaml
from datasets.colmap import Dataset, Parser
from datasets.colmap_rgba import DatasetRGBA
from easydict import EasyDict as edict
from simple_trainer_rgba import Runner

import imageio.v3 as iio

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy


# PyTorch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Experiment values & configs
EXPERIMENTS = {
    "rgba": [
        "mcmc_random_bkgd",
        "mcmc_alpha_loss",
        "default_random_bkgd",
        "default_alpha_loss",
    ],
    "rgb": ["default", "mcmc"],
}

# Argument parser
parser = argparse.ArgumentParser(description="Render test views")
parser.add_argument(
    "--data-type", type=str, default="rgba", choices=["rgba", "rgb"], help="Data type"
)
parser.add_argument(
    "--experiment", type=str, default="mcmc_random_bkgd", help="Experiment name"
)
parser.add_argument("--frame", type=str, default="00000000", help="Frame number")

args = parser.parse_args()

DATA_TYPE = args.data_type
EXPERIMENT = args.experiment

# Validate data type and experiment combination
if DATA_TYPE not in EXPERIMENTS:
    raise ValueError(
        f"Invalid data type: {DATA_TYPE}. Must be one of {list(EXPERIMENTS.keys())}"
    )

if EXPERIMENT not in EXPERIMENTS[DATA_TYPE]:
    raise ValueError(
        f"Invalid experiment: {EXPERIMENT} for data type: {DATA_TYPE}. Must be one of {EXPERIMENTS[DATA_TYPE]}"
    )

FRAME = args.frame
RESULT_DIR = f"/home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/{FRAME}/{DATA_TYPE}/{EXPERIMENT}"

# Config path
RESULT_CFG_PATH = os.path.join(RESULT_DIR, "cfg.yml")
with open(RESULT_CFG_PATH, "r") as f:
    cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

cfg = edict(cfg)


# Test dataloader
test_data_dir = f"/home/minhtran/Code/data/vocap/minh_2/frames/{FRAME}/test/{DATA_TYPE}"
test_parser = Parser(
    data_dir=test_data_dir,
    factor=cfg.data_factor,
    normalize=cfg.normalize_world_space,
    test_every=1,
)
if DATA_TYPE == "rgba":
    test_dataset = DatasetRGBA(test_parser, split="eval")
else:
    test_dataset = Dataset(test_parser, split="eval")
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=7,
    shuffle=False,
    num_workers=0,
)


# Runner
runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)


# Load checkpoint
ckpt_path = os.path.join(RESULT_DIR, "ckpts", "ckpt_29999_rank0.pt")
ckpt = torch.load(ckpt_path, map_location=device)
splats = ckpt["splats"]
runner.splats = splats


# Render test views
data = next(iter(test_loader))
camtoworlds = data["camtoworld"].to(device)
Ks = data["K"].to(device)
pixels = data["image"].to(device) / 255.0  # [1, H, W, 4]
pixels_alpha = pixels[..., 3:]
pixels = pixels[..., :3] * pixels_alpha  # Alpha blend for RGBA
masks = data["mask"].to(device) if "mask" in data else None
height, width = pixels.shape[1:3]

colors, alphas, _ = runner.rasterize_splats(
    camtoworlds=camtoworlds,
    Ks=Ks,
    width=width,
    height=height,
    sh_degree=cfg.sh_degree,
    near_plane=cfg.near_plane,
    far_plane=cfg.far_plane,
    masks=masks,
)
colors = torch.clamp(colors, 0.0, 1.0)

# Concatenate renders and alphas to get RGBA
renders_rgba = torch.cat([colors, alphas], dim=-1)
renders_rgba_np = renders_rgba.cpu().detach().numpy()
renders_rgba_np = (renders_rgba_np * 255.0).astype("uint8")


# Save rendered images
save_dir = os.path.join(test_data_dir, "gsplat_renders", EXPERIMENT)
os.makedirs(save_dir, exist_ok=True)
camera_ids = test_parser.camera_ids
for i in range(len(camera_ids)):
    camera_id = camera_ids[i]
    print(f"Saving rendered image for camera {camera_id}...")
    img_name = f"{FRAME}_{camera_id:03d}.png"
    render_rgba = renders_rgba_np[i, ...]
    iio.imwrite(os.path.join(save_dir, img_name), render_rgba)
