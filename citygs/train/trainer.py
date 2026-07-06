"""Shared training engine for the coarse and per-block stages.

A deliberately small subset of gsplat's ``simple_trainer``: 3DGS with
either the Default (densification) or MCMC (fixed budget) strategy,
L1 + SSIM (+ optional sparse-depth) losses, antialiased rasterization,
and gsplat's Grendel-style ``distributed=True`` when world_size > 1.
No pose/appearance optimization, no viewer thread — blocks train
headless in worker processes.
"""

import json
import math
import os
import time
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.losses import (
    depth_l1_loss,
    l1_loss,
    opacity_reg_loss,
    scale_reg_loss,
    ssim_loss,
)
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from ..ckpt import save_ckpt
from ..config import TrainConfig
from ..utils import set_random_seed

Strategy = Union[DefaultStrategy, MCMCStrategy]


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        splats_init: Dict[str, Tensor],
        strategy: Strategy,
        trainset,
        valset,
        result_dir: str,
        scene_scale: float,
        device: str = "cuda",
        world_rank: int = 0,
        world_size: int = 1,
    ):
        set_random_seed(42 + world_rank)
        self.cfg = cfg
        self.strategy = strategy
        self.trainset = trainset
        self.valset = valset
        self.result_dir = result_dir
        self.scene_scale = scene_scale
        self.device = device
        self.world_rank = world_rank
        self.world_size = world_size

        if cfg.depth_loss and cfg.batch_size != 1:
            raise ValueError("depth_loss requires batch_size=1 (ragged point lists)")

        os.makedirs(result_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(result_dir, "ckpts")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Parameters + optimizers with the Grendel batch-size LR scaling
        # rule: lr * sqrt(BS), betas/eps adjusted accordingly.
        lrs = {
            "means": cfg.means_lr * scene_scale,
            "scales": cfg.scales_lr,
            "quats": cfg.quats_lr,
            "opacities": cfg.opacities_lr,
            "sh0": cfg.sh0_lr,
            "shN": cfg.shN_lr,
        }
        self.splats = torch.nn.ParameterDict(
            {
                k: torch.nn.Parameter(v.to(device).float())
                for k, v in splats_init.items()
            }
        )
        BS = cfg.batch_size * world_size
        self.optimizers = {
            name: torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
            for name, lr in lrs.items()
        }

        self.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(strategy, DefaultStrategy):
            self.strategy_state = strategy.initialize_state(scene_scale=scene_scale)
        else:
            self.strategy_state = strategy.initialize_state()

        self.writer = None
        if world_rank == 0 and cfg.tb_every > 0:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=os.path.join(result_dir, "tb"))
            except ImportError:
                pass

    # ------------------------------------------------------------- render
    def rasterize(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        sh_degree: int,
        render_mode: str = "RGB",
    ):
        return rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=torch.cat([self.splats["sh0"], self.splats["shN"]], 1),
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            packed=self.cfg.packed,
            absgrad=(
                self.strategy.absgrad
                if isinstance(self.strategy, DefaultStrategy)
                else False
            ),
            rasterize_mode="antialiased" if self.cfg.antialiased else "classic",
            distributed=self.world_size > 1,
            render_mode=render_mode,
        )

    # -------------------------------------------------------------- train
    def train(self, max_steps: Optional[int] = None, ckpt_meta: Optional[dict] = None):
        cfg = self.cfg
        device = self.device
        max_steps = max_steps or cfg.max_steps
        ckpt_meta = ckpt_meta or {}

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            )
        ]
        loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
            pin_memory=True,
            drop_last=True,
        )
        loader_iter = iter(loader)

        tic = time.time()
        for step in range(max_steps):
            try:
                data = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                data = next(loader_iter)

            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            renders, alphas, info = self.rasterize(
                camtoworlds,
                Ks,
                width,
                height,
                sh_degree,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., :3], renders[..., 3:4]
            else:
                colors, depths = renders, None
            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            l1 = l1_loss(colors, pixels).mean()
            ssim = ssim_loss(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
            loss = torch.lerp(l1, ssim, cfg.ssim_lambda)
            depthloss = None
            if cfg.depth_loss and data["points"].shape[1] > 0:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]
                grid = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                ).unsqueeze(
                    2
                )  # [1, M, 1, 2]
                sampled = (
                    F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)
                    .squeeze(3)
                    .squeeze(1)
                )  # [1, M]
                depthloss = depth_l1_loss(
                    sampled, depths_gt, scene_scale=self.scene_scale
                )
                loss = loss + depthloss * cfg.depth_lambda
            if cfg.opacity_reg > 0.0:
                loss = loss + cfg.opacity_reg * opacity_reg_loss(
                    self.splats["opacities"]
                )
            if cfg.scale_reg > 0.0:
                loss = loss + cfg.scale_reg * scale_reg_loss(self.splats["scales"])

            loss.backward()

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            if isinstance(self.strategy, DefaultStrategy):
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            else:  # MCMC
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )

            if self.writer is not None and step % cfg.tb_every == 0:
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                if depthloss is not None:
                    self.writer.add_scalar("train/depth", depthloss.item(), step)

            if step % 500 == 0 and self.world_rank == 0:
                elapsed = time.time() - tic
                print(
                    f"[train] step {step}/{max_steps} loss={loss.item():.4f} "
                    f"n_gs={len(self.splats['means']):,} ({elapsed:.0f}s)",
                    flush=True,
                )

            if (step + 1) in cfg.save_steps or step == max_steps - 1:
                self.save(step, ckpt_meta)
            if (step + 1) in cfg.eval_steps or step == max_steps - 1:
                self.eval(step)

    # --------------------------------------------------------------- eval
    @torch.no_grad()
    def eval(self, step: int):
        if self.valset is None or len(self.valset) == 0:
            return
        cfg = self.cfg
        device = self.device
        loader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        psnrs, ssims = [], []
        for data in loader:
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            colors, _, _ = self.rasterize(camtoworlds, Ks, width, height, cfg.sh_degree)
            colors = colors[..., :3].clamp(0.0, 1.0)
            mse = F.mse_loss(colors, pixels)
            psnrs.append((10.0 * torch.log10(1.0 / mse)).item())
            ssims.append(
                1.0
                - ssim_loss(
                    colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
                ).item()
            )
        stats = {
            "step": step,
            "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)),
            "num_gaussians": len(self.splats["means"]),
        }
        if self.world_rank == 0:
            print(f"[eval] {stats}")
            with open(os.path.join(self.result_dir, f"val_step{step}.json"), "w") as f:
                json.dump(stats, f, indent=2)

    # --------------------------------------------------------------- save
    def save(self, step: int, meta: dict):
        meta = {**meta, "world_size": self.world_size, "rank": self.world_rank}
        name = f"ckpt_rank{self.world_rank}.pt" if self.world_size > 1 else "ckpt.pt"
        save_ckpt(os.path.join(self.ckpt_dir, name), self.splats, step, meta)
