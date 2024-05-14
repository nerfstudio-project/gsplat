import argparse
import json
import os
import random
import time
from typing import Dict, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.colmap import Dataset
from datasets.traj import generate_interpolated_path
from nerfview import CameraState, ViewerServer, view_lock
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gsplat.rendering_v2 import rasterization


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def _knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, algorithm="auto", metric="euclidean").fit(
        x_np
    )
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def _rgb_to_sh(rgb):
    C0 = 0.28209479177387814
    return rgb / C0


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_splats_with_optimizers(
    points: Tensor,  # [N, 3]
    rgbs: Tensor,  # [N, 3]
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    init_scale: Optional[float] = None,
    sparse_grad: bool = False,
    device: str = "cuda",
) -> torch.nn.ParameterDict:
    N = points.shape[0]

    if init_scale is None:
        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (_knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        scales = torch.log(torch.sqrt(dist2_avg)).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    else:
        scales = torch.log(torch.full((N, 3), init_scale))  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    # Initialize the GS color
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = _rgb_to_sh(rgbs)

    splats = torch.nn.ParameterDict(
        {
            "means3d": torch.nn.Parameter(points),  # [N, 3]
            "scales": torch.nn.Parameter(scales),  # [N, 3]
            "quats": torch.nn.Parameter(quats),  # [N, 4]
            "opacities": torch.nn.Parameter(opacities),  # [N,]
            "sh0": torch.nn.Parameter(colors[:, :1, :]),  # [N, 1, 3]
            "shN": torch.nn.Parameter(colors[:, 1:, :]),  # [N, K-1, 3]
        }
    ).to(device)
    optimizers = [
        # gsplat2 supports sparse gradients for these parameters
        (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [
                {"params": splats["means3d"], "name": "means3d"},  # has a lr schedule
                {"params": splats["scales"], "lr": 5e-3, "name": "scales"},
            ],
            eps=1e-15,
            # fused=True, TODO: benchmark fused optimizer
        ),
        torch.optim.Adam(
            [
                {"params": splats["quats"], "lr": 1e-3, "name": "quats"},
                {"params": splats["opacities"], "lr": 5e-2, "name": "opacities"},
                {"params": splats["sh0"], "lr": 2.5e-3, "name": "sh0"},
                {"params": splats["shN"], "lr": 2.5e-3 / 20, "name": "shN"},
            ],
            eps=1e-15,
            # fused=True,
        ),
    ]
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(self, args: argparse.Namespace) -> None:
        _set_random_seed(42)

        self.args = args
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(args.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{args.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{args.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{args.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Load data: Training data should contain initial points and colors.
        self.trainset = Dataset(
            data_dir=args.data_dir,
            factor=args.data_factor,
            normalize=True,
            split="train",
        )
        self.valset = Dataset(
            data_dir=args.data_dir,
            factor=args.data_factor,
            normalize=True,
            split="val",
        )
        self.scene_scale = self.trainset.scene_scale * 1.1
        print("Scene scale:", self.scene_scale)

        # Model
        self.splats, self.optimizers = create_splats_with_optimizers(
            torch.from_numpy(self.trainset.points).float(),
            torch.from_numpy(self.trainset.points_rgb / 255.0).float(),
            sh_degree=args.sh_degree,
            init_opacity=args.init_opa,
            init_scale=args.init_scale,
            sparse_grad=args.sparse_grad,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means3d"]))

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        self.server = ViewerServer(port=args.port, render_fn=self._viewer_render_fn)

        # Running stats for prunning & growing.
        n_gauss = len(self.splats["means3d"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
        }

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means3d"]  # [N, 3]
        # TODO: F.normalize is preventing sparse grad on quats!! Fuse it to CUDA kernel.
        quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.args.packed,
            sparse_grad=self.args.sparse_grad,
            **kwargs,
        )
        return render_colors, render_alphas, info

    # @line_profiler.profile
    def train(self):
        args = self.args
        device = self.device

        # Dump args.
        with open(f"{args.result_dir}/args.json", "w") as f:
            json.dump(vars(args), f)

        max_steps = args.max_steps
        init_step = 0

        lr_func = get_expon_lr_func(
            lr_init=0.00016 * self.scene_scale,
            lr_final=0.0000016 * self.scene_scale,
            lr_delay_mult=0.01,
            max_steps=30_000,
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
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            while self.server.training_state == "paused":
                time.sleep(0.01)
            view_lock.acquire()

            # Means3d has a learning rate schedule.
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    if param_group["name"] == "means3d":
                        param_group["lr"] = lr_func(step)
                        break
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
            sh_degree_to_use = min(step // args.sh_degree_interval, args.sh_degree)

            # forward
            colors, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=args.near_plane,
                far_plane=args.far_plane,
            )
            info["means2d"].retain_grad()  # used for running stats

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - args.ssim_lambda) + ssimloss * args.ssim_lambda
            loss.backward()
            pbar.set_description(
                f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}|"
            )

            # update running stats for prunning & growing
            if step < args.refine_stop_iter:
                self.update_running_stats(info)

                if step > args.refine_start_iter and step % args.refine_every == 0:
                    grads = self.running_stats["grad2d"] / self.running_stats[
                        "count"
                    ].clamp_min(1)

                    # grow GSs
                    is_grad_high = grads >= args.grow_grad2d
                    is_small = (
                        torch.exp(self.splats["scales"]).max(dim=-1).values
                        <= args.grow_scale3d * self.scene_scale
                    )
                    is_dupli = is_grad_high & is_small
                    n_dupli = is_dupli.sum().item()
                    self.refine_duplicate(is_dupli)

                    is_split = is_grad_high & ~is_small
                    is_split = torch.cat(
                        [
                            is_split,
                            # new GSs added by duplication will not be split
                            torch.zeros(n_dupli, device=device, dtype=torch.bool),
                        ]
                    )
                    n_split = is_split.sum().item()
                    self.refine_split(is_split)
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # prune GSs
                    is_prune = torch.sigmoid(self.splats["opacities"]) < args.prune_opa
                    if step > args.reset_every:
                        # The official code also implements sreen-size pruning but
                        # it's actually not being used due to a bug:
                        # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                        is_too_big = (
                            torch.exp(self.splats["scales"]).max(dim=-1).values
                            > 0.1 * self.scene_scale
                        )
                        is_prune = is_prune | is_too_big
                    n_prune = is_prune.sum().item()
                    self.refine_keep(~is_prune)
                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # reset running stats
                    self.running_stats["grad2d"].zero_()
                    self.running_stats["count"].zero_()

                if step % args.reset_every == 0:
                    self.reset_opa()

            # optimize
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # eval the full set
            if step in [i - 1 for i in args.eval_steps]:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Step {step}: Memory: {mem:.2f}GB")
                self.eval(step)
                self.render_traj()
                # save ckpt
                torch.save(self.splats.state_dict(), f"{self.ckpt_dir}/ckpt_{step}.pt")

            view_lock.release()

    @torch.no_grad()
    def update_running_stats(self, info: Dict):
        """Update running stats."""
        # normalize grads to [-1, 1] screen space
        grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0
        grads[..., 1] *= info["height"] / 2.0
        if self.args.packed:
            # grads is [nnz, 2]
            gs_ids = info["cindices"]  # [nnz] or None
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.running_stats["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids))
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )

    @torch.no_grad()
    def reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.splats["opacities"], max=torch.logit(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "opacities":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(opacities)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_split(self, mask: Tensor):
        """Utility function to grow GSs."""
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.splats["scales"][sel])  # [N, 3]
        quats = F.normalize(self.splats["quats"][sel], dim=-1)  # [N, 4]
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
                else:
                    repeats = [2] + [1] * (p.dim() - 1)
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new)
                # update optimizer
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key == "step":
                        continue
                    v = p_state[key]
                    # new params are assigned with zero optimizer states
                    # (worth investigating it)
                    v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                    p_state[key] = torch.cat([v[rest], v_split])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if v is None:
                continue
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            self.running_stats[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
        """Unility function to duplicate GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        args = self.args
        device = self.device

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
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=args.sh_degree,
                near_plane=args.near_plane,
                far_plane=args.far_plane,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.3f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means3d'])}"
        )
        # save to stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means3d"]),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)

    @torch.no_grad()
    def render_traj(self):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        args = self.args
        device = self.device

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
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=args.sh_degree,
                near_plane=args.near_plane,
                far_plane=args.far_plane,
                render_mode="RGB+D",
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
        video_dir = f"{args.result_dir}/videos/"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj.mp4")

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

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.args.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


if __name__ == "__main__":
    """
    python simple_trainer.py --data_dir data/progressive/university1/ --data_factor 1 \
        --grow_scale3d 0.001 --result_dir  results/university1 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/360_v2/garden",
        help="Path to the MipNeRF360 dataset",
    )
    parser.add_argument(
        "--data_factor",
        type=int,
        default=4,
        help="Downsample factor for the dataset",
    )
    parser.add_argument(
        "--result_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for the viewer server"
    )
    parser.add_argument(
        "--max_steps", type=int, default=30_000, help="Number of training steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        nargs="+",
        default=[7_000, 30_000],
        help="Steps to evaluate the model",
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="Degree of spherical harmonics"
    )
    parser.add_argument(
        "--init_opa", type=float, default=0.1, help="Initial opacity of GS"
    )
    parser.add_argument(
        "--init_scale", type=float, default=None, help="Initial scale of GS"
    )
    parser.add_argument(
        "--ssim_lambda", type=float, default=0.2, help="Weight for SSIM loss"
    )
    parser.add_argument(
        "--sh_degree_interval",
        type=int,
        default=1000,
        help="Turn on another SH degree every this steps",
    )
    parser.add_argument(
        "--near_plane", type=float, default=0.01, help="Near plane clipping distance"
    )
    parser.add_argument(
        "--far_plane", type=float, default=1e10, help="Far plane clipping distance"
    )
    parser.add_argument(
        "--prune_opa",
        type=float,
        default=0.005,
        help="GSs with opacity below this value will be pruned",
    )
    parser.add_argument(
        "--grow_grad2d",
        type=float,
        default=0.0002,
        help="GSs with image plane gradient above this value will be split/duplicated",
    )
    parser.add_argument(
        "--grow_scale3d",
        type=float,
        default=0.01,
        help="GSs with scale below this value will be duplicated. Above will be split",
    )
    parser.add_argument(
        "--refine_start_iter",
        type=int,
        default=500,
        help="Start refining GSs after this iteration",
    )
    parser.add_argument(
        "--refine_stop_iter",
        type=int,
        default=15_000,
        help="Stop refining GSs after this iteration",
    )
    parser.add_argument(
        "--reset_every",
        type=int,
        default=3000,
        help="Reset opacities every this steps",
    )
    parser.add_argument(
        "--refine_every",
        type=int,
        default=100,
        help="Refine GSs every this steps",
    )
    parser.add_argument(
        "--packed",
        action="store_true",
        help="Use packed mode for rasterization (less memory usage but slightly slower)",
    )
    parser.add_argument(
        "--sparse_grad",
        action="store_true",
        help="Use sparse gradients for optimization",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the .pt file. If provide, it will skip training and render a video",
    )
    args = parser.parse_args()
    if args.sparse_grad:
        assert args.packed, "Sparse gradients require packed mode"

    runner = Runner(args)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt[k]
        runner.render_traj()
    else:
        runner.train()

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(1000000)
