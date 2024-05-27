import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from datasets.colmap import Dataset
from datasets.traj import generate_interpolated_path
from model import (
    Gaussians,
    refine_duplicate,
    refine_keep,
    refine_split,
    reset_opac,
    update_lr,
)
from nerfview import CameraState, ViewerServer, view_lock
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import get_expon_lr_func, set_random_seed


@dataclass
class Config:
    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"

    # Port for the viewer server
    port: int = 8080

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opac: float = 0.1
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # GSs with opacity below this value will be pruned
    prune_opac: float = 0.005
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100

    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None
    # Do not wait for the viewer to exit
    exit_once_done: bool = False

    # Use packed mode for rasterization (less memory usage but slightly slower)
    packed: bool = False


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Load data: Training data should contain initial points and colors.
        self.trainset = Dataset(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            split="train",
        )
        self.valset = Dataset(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            split="val",
        )
        self.scene_scale = self.trainset.scene_scale * 1.1
        print("Scene scale:", self.scene_scale)

        # Model
        if cfg.ckpt is not None:
            ckpt = torch.load(cfg.ckpt, map_location=self.device)
            self.model = Gaussians.from_state_dict(ckpt["params"]).to(self.device)
            self.step = ckpt["step"]
        else:
            self.model = Gaussians.from_pointcloud(
                points=torch.from_numpy(self.trainset.points).float(),
                rgbs=torch.from_numpy(self.trainset.points_rgb / 255.0).float(),
                sh_degree=cfg.sh_degree,
                init_opac=cfg.init_opac,
            ).to(self.device)
            self.step = 0
        print("Model initialized. Number of GS:", len(self.model.params["means3d"]))

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        self.server = ViewerServer(port=cfg.port, render_fn=self._viewer_render_fn)

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps

        optimizers = [
            torch.optim.Adam(
                [{"params": v, "name": k} for k, v in self.model.params.items()],
                eps=1e-15,
                # fused=False,  # TODO: benchmark fused optimizer, might be faster?
            ),
        ]

        # Running stats for prunning & growing.
        self.model.running_stats = {
            "grad2d": torch.zeros(len(self.model), device=self.device),
            "count": torch.zeros(len(self.model), device=self.device, dtype=torch.int),
        }

        lr_func = get_expon_lr_func(
            lr_init=0.00016 * self.scene_scale,
            lr_final=0.0000016 * self.scene_scale,
            lr_delay_mult=0.01,
            max_steps=30_000,  # hard-code to 30_000
        )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        tic = time.time()
        pbar = tqdm.tqdm(range(self.step, max_steps))
        for step in pbar:
            self.step = step
            while self.server.training_state == "paused":
                time.sleep(0.01)
            view_lock.acquire()

            # Means3d has a learning rate schedule.
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    if param_group["name"] == "means3d":
                        param_group["lr"] = lr_func(step)
                        break

            # learning rate schedule
            update_lr(
                optimizers,
                lrs={
                    "means3d": lr_func(step),
                    "scales": 5e-3,
                    "quats": 1e-3,
                    "opacities": 5e-2,
                    "sh0": 2.5e-3,
                    "shN": 2.5e-3 / 20,
                },
            )

            # fetch data
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            height, width = pixels.shape[1:3]

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, meta = self.model(
                viewmats=torch.linalg.inv(camtoworlds),
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                packed=cfg.packed,
            )
            meta["means2d"].retain_grad()  # used for running stats

            # loss
            l1loss = F.l1_loss(renders, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), renders.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            loss.backward()
            pbar.set_description(
                f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}|"
            )

            # update running stats for prunning & growing
            if step < cfg.refine_stop_iter:
                self.update_running_stats(meta)

                if step > cfg.refine_start_iter and step % cfg.refine_every == 0:
                    grads = self.model.running_stats[
                        "grad2d"
                    ] / self.model.running_stats["count"].clamp_min(1)

                    # grow GSs
                    is_grad_high = grads >= cfg.grow_grad2d
                    is_small = (
                        torch.exp(self.model.params["scales"]).max(dim=-1).values
                        <= cfg.grow_scale3d * self.scene_scale
                    )
                    is_dupli = is_grad_high & is_small
                    n_dupli = is_dupli.sum().item()
                    refine_duplicate(self.model, optimizers, is_dupli)

                    is_split = is_grad_high & ~is_small
                    is_split = torch.cat(
                        [
                            is_split,
                            # new GSs added by duplication will not be split
                            torch.zeros(n_dupli, device=device, dtype=torch.bool),
                        ]
                    )
                    n_split = is_split.sum().item()
                    refine_split(self.model, optimizers, is_split)
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(self.model)} GSs."
                    )

                    # prune GSs
                    is_prune = (
                        torch.sigmoid(self.model.params["opacities"]) < cfg.prune_opac
                    )
                    if step > cfg.reset_every:
                        # The official code also implements sreen-size pruning but
                        # it's actually not being used due to a bug:
                        # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                        is_too_big = (
                            torch.exp(self.model.params["scales"]).max(dim=-1).values
                            > 0.1 * self.scene_scale
                        )
                        is_prune = is_prune | is_too_big
                    n_prune = is_prune.sum().item()
                    refine_keep(self.model, optimizers, ~is_prune)
                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(self.model.params['means3d'])} GSs."
                    )

                    # reset running stats
                    self.model.running_stats["grad2d"].zero_()
                    self.model.running_stats["count"].zero_()

                if step % cfg.reset_every == 0:
                    reset_opac(self.model, optimizers)

            # optimize
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval()
                self.render_traj()

            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                toc = time.time()
                stats = {
                    "mem": mem,
                    "ellipse_time": toc - tic,
                    "num_GS": len(self.model),
                }
                print("Step: ", step, "stats: ", stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "params": self.model.params.state_dict(),
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            view_lock.release()

    @torch.no_grad()
    def update_running_stats(self, meta: Dict):
        """Update running stats."""
        # normalize grads to [-1, 1] screen space
        grads = meta["means2d"].grad.clone()
        grads[..., 0] *= meta["width"] / 2.0
        grads[..., 1] *= meta["height"] / 2.0

        if self.cfg.packed:
            # grads is [nnz, 2]
            gs_ids = meta["cindices"]  # [nnz]
            self.model.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.model.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids)
            )
        else:
            # grads is [C, N, 2]
            sel = meta["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.model.running_stats["grad2d"].index_add_(
                0, gs_ids, grads[sel].norm(dim=-1)
            )
            self.model.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )

    @torch.no_grad()
    def eval(self):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        step = self.step

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            renders, _, _ = self.model(
                viewmats=torch.linalg.inv(camtoworlds),
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                packed=cfg.packed,
            )  # [1, H, W, 3]
            renders = torch.clamp(renders, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([pixels, renders], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            renders = renders.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(renders, pixels))
            metrics["ssim"].append(self.ssim(renders, pixels))
            metrics["lpips"].append(self.lpips(renders, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.model)}"
        )
        # save to stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.model),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)

    @torch.no_grad()
    def render_traj(self):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        step = self.step

        camtoworlds = self.trainset.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(self.trainset.K).float().to(device)
        width = self.trainset.width
        height = self.trainset.height

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _ = self.model(
                viewmats=torch.linalg.inv(camtoworlds[i : i + 1]),
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                packed=cfg.packed,
                # Render RGB and expected depth at once.
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos/"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(self, camera_state: CameraState, img_wh: Tuple[int, int]):
        """Callable function for the viewer."""
        fov = camera_state.fov
        c2w = camera_state.c2w
        W, H = img_wh

        focal_length = H / 2.0 / np.tan(fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.model(
            viewmats=torch.linalg.inv(c2w[None]),
            Ks=K[None],
            width=W,
            height=H,
            packed=self.cfg.packed,
            # skip GSs that have small image radius (in pixels)
            radius_clip=3.0,
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        runner.eval()
        runner.render_traj()
    else:
        runner.train()

    if not cfg.exit_once_done:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
