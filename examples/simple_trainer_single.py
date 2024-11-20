import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam
from iresnet import iResNet

from utils_cubemap import homogenize, dehomogenize, fov2focal, focal2fov, generate_pts, init_from_coeff, plot_points, init_from_colmap, generate_control_pts, getProjectionMatrix, center_crop, apply_distortion, rotate_camera, generate_pts_up_down_left_right, interpolate_with_control, apply_flow_up_down_left_right, mask_half, render_cubemap, generate_circular_mask
from utils_mitsuba import fetchPly
from utils_mitsuba import readCamerasFromTransforms, PILtoTorch, cubemap_to_panorama, rotation_matrix_to_quaternion, quaternion_to_rotation_matrix
from datasets.normalize import transform_cameras, transform_points

@dataclass
class Config:
    # ablation study
    control_point_sample_scale: int = 32
    control_point_sample_scale_eval: int = 4
    iresnet_lr: float = 1e-7
    step2start: int = 3000
    step2end: int = 7000
    scale_x: float = 2
    scale_y: float = 2
    scale_x_eval: float = 2
    scale_y_eval: float = 2
    when2addmask: int = 100
    update_min_opacity: float = 0.005
    iresnet_path: str = ''


    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
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

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [1, 7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_011, 10_011, 20_111, 29_996])

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
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

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

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
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


def create_splats_with_optimizers(
    parser: Parser,
    pcd,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    elif init_type == 'netflix':
        points = torch.tensor(np.asarray(pcd.points)).float().cuda()
        points = transform_points(parser.transform, points.cpu()).float().cuda()
        rgbs = torch.tensor(np.asarray(pcd.colors)).float().cuda()
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers

class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
        , evaluation: bool    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(f"{cfg.result_dir}/monitor", exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        pcd = None
        if cfg.init_type == 'netflix':
            ply_path = os.path.join(cfg.data_dir, f'fish/sparse/0/fused.ply')
            pcd = fetchPly(ply_path)

        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            pcd,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        if world_rank == 0:
            self.lens_net = iResNet().cuda()
            l_lens_net = [{'params': self.lens_net.parameters(), 'lr': 1e-5}]
            optimizer_lens_net = torch.optim.Adam(l_lens_net, eps=1e-15)
            scheduler_lens_net = torch.optim.lr_scheduler.MultiStepLR(optimizer_lens_net, milestones=[5000], gamma=0.5)

            viewpoint_cam = self.trainset.__getitem__(0)
            coeff = self.trainset.__getitem__(0)['coeff']
            if not evaluation:
                init_from_colmap(viewpoint_cam, coeff, cfg.result_dir, self.lens_net, optimizer_lens_net, scheduler_lens_net, iresnet_lr=1e-12)
        else:
            self.lens_net = iResNet().cuda()

        if world_size > 1:
            dist.barrier()
            for param in self.lens_net.state_dict().values():
                dist.broadcast(param, src=0)

            l_lens_net = [{'params': self.lens_net.parameters(), 'lr': cfg.iresnet_lr}]
            self.optimizer_lens_net = torch.optim.Adam(l_lens_net, eps=1e-15)
            self.lens_net = DDP(self.lens_net)
        else:
            l_lens_net = [{'params': self.lens_net.parameters(), 'lr': cfg.iresnet_lr}]
            self.optimizer_lens_net = torch.optim.Adam(l_lens_net, eps=1e-15)

        self.intrinsic_adjust = torch.nn.Parameter(torch.zeros(len(self.trainset), 2, device='cuda'))
        self.intrinsic_optimizers = torch.optim.Adam([self.intrinsic_adjust], lr= 1e-5 * math.sqrt(cfg.batch_size), weight_decay=1e-6,)
        if world_size > 1:
            self.pose_adjust = DDP(self.pose_adjust)

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
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
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)
        self.trainloader_iter = trainloader_iter

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        debug_grid = None
        #debug_grid = torch.load('/home/yd428/gsplat/examples/results/netflix/paul_garden_scale2/P_view_outsidelens_direction.pt')
        #debug_grid = torch.load('/home/yd428/gsplat/examples/results/nyc/P_view_outsidelens_direction.pt')
        #debug_grid = torch.load('/home/yd428/gsplat/examples/results/netflix/paul_office_crop_deblur_optpose/P_view_outsidelens_direction.pt')
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            Ks[0][0][0] += self.intrinsic_adjust[step % len(self.trainset)][0]
            Ks[0][1][1] += self.intrinsic_adjust[step % len(self.trainset)][1]
            Ks[0][0][-1] = Ks[0][0][-1] * cfg.scale_x
            Ks[0][1][-1] = Ks[0][1][-1] * cfg.scale_y
            renders, alphas, info = self.rasterize_splats(camtoworlds=camtoworlds,Ks=Ks,width=int(width*cfg.scale_x),height=int(height*cfg.scale_y),sh_degree=sh_degree_to_use,near_plane=cfg.near_plane,far_plane=cfg.far_plane,image_ids=image_ids,render_mode="RGB+ED" if cfg.depth_loss else "RGB",masks=masks)
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(int(height*cfg.scale_y), device=self.device) + 0.5) / int(height*cfg.scale_y),
                    (torch.arange(int(width*cfg.scale_x), device=self.device) + 0.5) / int(width*cfg.scale_x),
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # apply distortion field
            apply2gt = False
            if apply2gt:
                P_sensor, P_view_insidelens_direction = generate_control_pts(data, control_point_sample_scale=16, flow_scale=[1.5, 1.5])
                fovx_orig = focal2fov(Ks[0][0][0], 1.5*width)
                fovy_orig = focal2fov(Ks[0][1][1], 1.5*height)
                projection_matrix = getProjectionMatrix(cfg.near_plane, cfg.far_plane, fovx_orig, fovy_orig).cuda()
                fish_gt, mask, flow_apply2_gt_or_img = apply_distortion(self.lens_net, P_view_insidelens_direction, P_sensor, data, colors, projection_matrix, flow_scale=[2., 2.], apply2gt=apply2gt)
                fish_gt = fish_gt.unsqueeze(0)
                image = colors.permute(0, 3, 1, 2)
            else:
                P_sensor, P_view_insidelens_direction = generate_control_pts(data, control_point_sample_scale=cfg.control_point_sample_scale, flow_scale=[3., 3.])
                fovx = focal2fov(Ks[0][0][0], cfg.scale_x*width)
                fovy = focal2fov(Ks[0][1][1], cfg.scale_y*height)
                projection_matrix = getProjectionMatrix(cfg.near_plane, cfg.far_plane, fovx, fovy).cuda()
                if step == cfg.step2start:
                    for param_group in self.optimizer_lens_net.param_groups:
                        lr = param_group['lr']
                    print(lr)
                    debug_grid = None
                if step == cfg.step2end or step == 0:
                    image, mask, flow_apply2_gt_or_img, debug_grid = apply_distortion(self.lens_net, P_view_insidelens_direction, P_sensor, data, colors, projection_matrix, flow_scale=[3., 3.], debug_grid=debug_grid, update_debug_grid=True)
                else:
                    image, mask, flow_apply2_gt_or_img = apply_distortion(self.lens_net, P_view_insidelens_direction, P_sensor, data, colors, projection_matrix, flow_scale=[3., 3.], debug_grid=debug_grid)
                fish_gt = (data['fisheye_image'].permute(0, 3, 1, 2)/255).cuda()


            #torchvision.utils.save_image(colors.permute(0, 3, 1, 2), os.path.join(cfg.result_dir, f'render_pers.png'))
            #torchvision.utils.save_image(image, os.path.join(cfg.result_dir, f'render.png'))
            #torchvision.utils.save_image(fish_gt, os.path.join(cfg.result_dir, f'gt.png'))
            #torchvision.utils.save_image(data['image'].permute(0, 3, 1, 2)/255, os.path.join(cfg.result_dir, f'gt_pers.png'))
            #import pdb;pdb.set_trace()
            if step >= cfg.when2addmask:
                black_boundary_mask = (image != 0).float()
                fish_gt = fish_gt * black_boundary_mask

            # loss
            l1loss = F.l1_loss(image, fish_gt)
            ssimloss = 1.0 - fused_ssim(
                image, fish_gt, padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            if world_rank == 0 and step % 500 == 0:
                torchvision.utils.save_image(image, os.path.join(cfg.result_dir, f'monitor/perspective_{step}.png'))
                torchvision.utils.save_image(fish_gt, os.path.join(cfg.result_dir, f'monitor/gt_{step}.png'))


            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps]:
            #if step in [10001, 20001, 29998]:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
                torch.save(self.lens_net.state_dict(), f"{self.ckpt_dir}/lens_net_{step}.pt")

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()
            if debug_grid == None:
                self.optimizer_lens_net.step()
                self.optimizer_lens_net.zero_grad(set_to_none=True)
            #self.intrinsic_optimizers.step()
            #self.intrinsic_optimizers.zero_grad(set_to_none=True)

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            #if step in [1, 10000, 20000, 29999] and False:
            if step in [i - 1 for i in cfg.eval_steps]:
                with torch.no_grad():
                    fovx = focal2fov(Ks[0][0][0], cfg.scale_x_eval*width)
                    fovy = focal2fov(Ks[0][1][1], cfg.scale_y_eval*height)
                    projection_matrix = getProjectionMatrix(cfg.near_plane, cfg.far_plane, fovx, fovy).cuda()
                    flow_scale = [3., 3.]
                    P_sensor_, P_view_insidelens_direction_ = generate_control_pts(data, control_point_sample_scale=4, flow_scale=[3., 3.])
                    P_view_outsidelens_direction_ = self.lens_net.forward(P_view_insidelens_direction_, sensor_to_frustum=apply2gt)
                    camera_directions_w_lens_ = homogenize(P_view_outsidelens_direction_)
                    control_points_ = camera_directions_w_lens_.reshape((P_sensor_.shape[0], P_sensor_.shape[1], 3))[:, :, :2]
                    flow = control_points_ @ projection_matrix[:2, :2]
                    flow_eval = F.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(int(data['fisheye_image'].shape[1]*flow_scale[0]), int(data['fisheye_image'].shape[2]*flow_scale[1])), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
                self.eval(step, flow_eval, apply2gt, cfg.scale_x_eval, cfg.scale_y_eval)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, flow_apply2_gt_or_img: Tensor, apply2gt: bool, scale_x: float, scale_y: float, stage: str = "val"):

        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            Ks[0][0][-1] = Ks[0][0][-1] * scale_x
            Ks[0][1][-1] = Ks[0][1][-1] * scale_y
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=int(width*scale_x),
                height=int(height*scale_y),
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]

            torchvision.utils.save_image(colors.permute(0, 3, 1, 2), os.path.join(cfg.result_dir, f'renders/perspective_{i}.png'))
            if not apply2gt:
                colors = F.grid_sample(
                    colors.permute(0, 3, 1, 2),
                    flow_apply2_gt_or_img.unsqueeze(0),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                colors = center_crop(colors, data['fisheye_image'].shape[1], data['fisheye_image'].shape[2]).permute(0, 2, 3, 1)
            else:
                pixels = F.grid_sample(
                    data["fisheye_image"].cuda().permute(0, 3, 1, 2)/255,
                    flow_apply2_gt_or_img.unsqueeze(0),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).permute(0, 2, 3, 1)

            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            if not apply2gt:
                pixels = data["fisheye_image"].to(device) / 255.0
            mask = (~((colors[..., 0] == 0.0) & (colors[..., 1] == 0.0) & (colors[..., 2] == 0.0))).unsqueeze(0).float().permute(0, 2, 3, 1)
            pixels = pixels * mask
            colors = colors * mask
            canvas_list = [pixels, colors]
            torchvision.utils.save_image(pixels.permute(0, 3, 1, 2), os.path.join(cfg.result_dir, f'renders/gt_{i}.png'))
            torchvision.utils.save_image(colors.permute(0, 3, 1, 2), os.path.join(cfg.result_dir, f'renders/rendering_{i}.png'))
            if apply2gt: continue

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                black_boundary_mask = (pixels_p != 0).float()
                metrics["psnr"].append(self.psnr(colors_p, pixels_p*black_boundary_mask))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p*black_boundary_mask))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p*black_boundary_mask))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))

        if world_rank == 0 and not apply2gt:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")


    if cfg.ckpt is not None:
        runner = Runner(local_rank, world_rank, world_size, cfg, True)
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        runner.lens_net.load_state_dict(ckpts.pop(-1))
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        #runner.eval(step=step)
        #runner.render_traj(step=step)
        #if cfg.compression is not None:
        #    runner.run_compression(step=step)


        os.makedirs(f"{cfg.result_dir}/validation_{step}", exist_ok=True)
        os.makedirs(f"{cfg.result_dir}/validation_{step}/forward", exist_ok=True)
        os.makedirs(f"{cfg.result_dir}/validation_{step}/perspective", exist_ok=True)
        os.makedirs(f"{cfg.result_dir}/validation_{step}/fish", exist_ok=True)
        if 'cube' in cfg.result_dir:
            with open('/home/yd428/playaround_gaussian_platting/cvpr/cube_lr7_apply2render_res2/images.txt') as f:
                lines = f.readlines()
        elif 'rock' in cfg.result_dir:
            with open('/home/yd428/playaround_gaussian_platting/cvpr/rock_vanilla/images.txt') as f:
                lines = f.readlines()
        elif 'chairs' in cfg.result_dir:
            with open('/home/yd428/playaround_gaussian_platting/cvpr/chairs_vanilla/images.txt') as f:
                lines = f.readlines()
        elif 'flowers' in cfg.result_dir:
            with open('/home/yd428/playaround_gaussian_platting/cvpr/flowers_vanilla/images.txt') as f:
                lines = f.readlines()
        elif 'garden' in cfg.result_dir:
            with open('/home/yd428/playaround_gaussian_platting/netflix/netflix_garden_colmap_0.5/images.txt') as f:
                lines = f.readlines()
        elif 'room' in cfg.result_dir:
            with open('/home/yd428/playaround_gaussian_platting/netflix/netflix_other_room_lr8_apply2render_res2_scale2.5_/images.txt') as f:
                lines = f.readlines()

        with torch.no_grad():
            trainloader = torch.utils.data.DataLoader(
                runner.trainset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
            )
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)
            fish_gt = (data['fisheye_image'].permute(0, 3, 1, 2)/255).cuda()
            transform = data["transform"][0].float()
            #print(data["camtoworld"])
            #print(data["image_name"])
            #torchvision.utils.save_image(fish_gt, '/home/yd428/coor/new.png')
            #print(transform_cameras(, data["camtoworld"]))
            Ks = data["K"].cuda()
            height, width = Ks[0][1][-1]*2, Ks[0][0][-1]*2
            fovx = focal2fov(Ks[0][0][0], cfg.scale_x_eval*width)
            fovy = focal2fov(Ks[0][1][1], cfg.scale_y_eval*height)
            projection_matrix = getProjectionMatrix(cfg.near_plane, cfg.far_plane, fovx, fovy).cuda()
            flow_scale = [3., 3.]
            P_sensor_, P_view_insidelens_direction_ = generate_control_pts(data, control_point_sample_scale=cfg.control_point_sample_scale_eval, flow_scale=[3., 3.])
            P_view_outsidelens_direction_ = runner.lens_net.forward(P_view_insidelens_direction_, sensor_to_frustum=False)
            camera_directions_w_lens_ = homogenize(P_view_outsidelens_direction_)
            control_points_ = camera_directions_w_lens_.reshape((P_sensor_.shape[0], P_sensor_.shape[1], 3))[:, :, :2]
            flow = control_points_ @ projection_matrix[:2, :2]
            flow_apply2_gt_or_img = F.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(int(data['fisheye_image'].shape[1]*flow_scale[0]), int(data['fisheye_image'].shape[2]*flow_scale[1])), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)

            i = 0
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                image_name = parts[9]
                quaternion_tensor = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
                R = quaternion_to_rotation_matrix(quaternion_tensor).cuda()
                T = torch.tensor([tx, ty, tz]).cuda().view(-1, 1)
                Rt = torch.cat((R, T), dim=1)
                last_row = torch.tensor([[0., 0., 0., 1.]]).cuda()
                w2c = torch.cat((Rt, last_row), dim=0)
                camtoworlds = w2c.inverse().unsqueeze(0)
                camtoworlds = torch.from_numpy(transform_cameras(transform, camtoworlds.cpu())).cuda()

                if i == 0:
                    Ks[0][0][-1] = Ks[0][0][-1] * cfg.scale_x_eval
                    Ks[0][1][-1] = Ks[0][1][-1] * cfg.scale_y_eval
                image_ids = torch.tensor([0]).cuda()
                masks = None
                colors, _, _ = runner.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=int(width*cfg.scale_x_eval),
                    height=int(height*cfg.scale_y_eval),
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    masks=masks,
                )  # [1, H, W, 3]
                torchvision.utils.save_image(colors.permute(0, 3, 1, 2)[0], os.path.join(cfg.result_dir, f'validation_{step}/perspective/perspective_{i}.png'))
                colors = F.grid_sample(
                    colors.permute(0, 3, 1, 2),
                    flow_apply2_gt_or_img.unsqueeze(0),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                colors = center_crop(colors, data['fisheye_image'].shape[1], data['fisheye_image'].shape[2])
                torchvision.utils.save_image(colors[0], os.path.join(cfg.result_dir, f'validation_{step}/fish/perspective_{i}.png'))
                i += 1
                print(i)
    else:
        runner = Runner(local_rank, world_rank, world_size, cfg, False)
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    #cfg.strategy.min_opacity = cfg.update_min_opacity
    cfg.adjust_steps(cfg.steps_scaler)

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

    cli(main, cfg, verbose=True)
