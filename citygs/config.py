"""Configuration dataclasses for every pipeline stage.

All stage entrypoints are tyro CLIs over these dataclasses. The pipeline
orchestrator passes configs to subprocesses by serializing them to JSON
(``--config-json``), so the CLI surface and the programmatic surface stay
identical.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


def _to_jsonable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        }
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_config(cfg, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(cfg), f, indent=2)


def load_config(cls: Type[T], path: str) -> T:
    with open(path) as f:
        data = json.load(f)
    return _from_dict(cls, data)


def _from_dict(cls: Type[T], data: dict) -> T:
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = f.type if isinstance(f.type, type) else None
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            value = _from_dict(ftype, value)
        elif isinstance(value, dict) and dataclasses.is_dataclass(_resolve(f.type)):
            value = _from_dict(_resolve(f.type), value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _resolve(tp):
    """Resolve a possibly-stringified dataclass field type."""
    if isinstance(tp, type):
        return tp
    if isinstance(tp, str):
        return _KNOWN_TYPES.get(tp)
    return None


@dataclass
class TrainConfig:
    """Hyperparameters shared by the coarse and per-block trainers."""

    max_steps: int = 30_000
    batch_size: int = 1
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    ssim_lambda: float = 0.2
    init_opacity: float = 0.1
    init_scale: float = 1.0
    near_plane: float = 0.01
    far_plane: float = 1e10
    # Rasterization options. Antialiasing helps multi-resolution aerial captures.
    antialiased: bool = True
    packed: bool = False
    random_bkgd: bool = False

    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    # Sparse-depth supervision from the SfM points (needs points.npz indices).
    depth_loss: bool = False
    depth_lambda: float = 1e-2

    eval_steps: List[int] = field(default_factory=lambda: [30_000])
    save_steps: List[int] = field(default_factory=lambda: [30_000])
    num_workers: int = 4
    image_cache_size: int = 256
    tb_every: int = 100


@dataclass
class CoarseConfig:
    """Global coarse prior, trained with the MCMC strategy (fixed budget).

    Run single-GPU or sharded across all visible GPUs with
    ``distributed=True`` rasterization (Grendel-style). The Gaussian budget
    ``cap_max`` is the total across ranks; MCMC keeps per-rank counts
    balanced because relocation/additions happen per rank on its own shard.
    """

    manifest: str = "results/scene.json"
    result_dir: str = "results/coarse"
    train: TrainConfig = field(default_factory=TrainConfig)

    # Total number of Gaussians (across all ranks).
    cap_max: int = 8_000_000
    mcmc_noise_lr: float = 5e5
    mcmc_refine_start_iter: int = 500
    mcmc_refine_stop_iter: int = 25_000
    mcmc_refine_every: int = 100
    mcmc_min_opacity: float = 0.005
    # MCMC needs these regularizers to prune dead Gaussians.
    opacity_reg: float = 0.01
    scale_reg: float = 0.01

    # Extra downsampling on top of the manifest resolution, coarse-only.
    # The coarse prior does not need full resolution; this is also the
    # fallback that buys Gaussian budget without multi-GPU all-to-all.
    downsample: int = 4

    # Initialize from at most this many SfM points (subsampled if more).
    max_init_points: int = 4_000_000


@dataclass
class PartitionConfig:
    """Contracted-space grid partition + camera assignment + load balance."""

    manifest: str = "results/scene.json"
    coarse_ckpt: str = ""  # defaults to the manifest artifact entry
    result_dir: str = "results/partition"

    # Grid resolution in contracted space (x, y). z is never split:
    # city scenes are flat and vertical cuts hurt far more than they help.
    grid: Tuple[int, int] = (4, 4)
    # Robust foreground bounds: cameras within `mad_threshold` MADs of the
    # median are inliers; the box is their extent times `fg_margin`.
    mad_threshold: float = 6.0
    fg_margin: float = 1.05
    # Optional manual override of the foreground box (world space):
    # (cx, cy, cz, ex, ey, ez) center + half-extents. Empty = auto.
    fg_bounds: Tuple[float, ...] = ()

    # Camera assignment: a camera trains a block if the block's coarse
    # Gaussians cover at least this fraction of its image (cheap opacity x
    # projected-area proxy, no rendering).
    coverage_threshold: float = 0.05
    # Blocks are guaranteed at least this many training cameras (top-k).
    min_cameras: int = 25
    # Subsample each block to at most this many Gaussians for the proxy.
    max_proxy_gaussians: int = 100_000
    # Cameras processed per batch during assignment.
    camera_chunk: int = 128
    # Expand block bounds by this margin (in contracted units) when
    # initializing finetune from the coarse prior.
    block_margin: float = 0.1

    device: str = "auto"  # "auto" | "cuda" | "cpu"


@dataclass
class BlockConfig:
    """Finetune one block on one GPU."""

    manifest: str = "results/scene.json"
    block_id: int = 0
    coarse_ckpt: str = ""  # defaults to the manifest artifact entry
    result_dir: str = "results/blocks"
    train: TrainConfig = field(default_factory=TrainConfig)

    # Initialize from coarse Gaussians within the block cell expanded by
    # this margin (contracted units); the final model is cropped back to
    # the exact cell so merge stays a plain concat.
    init_margin: float = 0.1

    # DefaultStrategy (densification) settings; absgrad is the better
    # criterion for aerial scenes (see EXPLORATION.md).
    absgrad: bool = True
    grow_grad2d: float = 8e-4
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000

    # Scale finetune steps with the block's camera count:
    # steps = max_steps * sqrt(n_cams / median_cams), clamped to
    # [min_steps_factor, max_steps_factor] * max_steps. 0 median = disabled.
    adaptive_steps: bool = True
    min_steps_factor: float = 0.5
    max_steps_factor: float = 1.5


@dataclass
class MergeConfig:
    """Crop every block to its own cell and concatenate."""

    manifest: str = "results/scene.json"
    blocks_dir: str = "results/blocks"
    result_dir: str = "results/merged"
    save_ply: bool = True


@dataclass
class EvalConfig:
    """Render the val split with the merged model and report PSNR/SSIM."""

    manifest: str = "results/scene.json"
    ckpt: str = "results/merged/merged.pt"
    result_dir: str = "results/eval"
    save_images: bool = False
    lpips: bool = False  # needs torchmetrics[image] weights download


@dataclass
class PipelineConfig:
    """End-to-end orchestration. Stages skip themselves when their output
    artifacts already exist; pass ``force`` to redo from a stage onwards."""

    data_dir: str = "data/scene"
    result_dir: str = "results/scene"
    data_factor: int = 1
    test_every: int = 8

    # GPUs used by the pipeline (indices). Coarse uses all of them when
    # coarse_distributed, the block queue runs one block per GPU.
    gpus: List[int] = field(default_factory=lambda: [0])
    # Grendel-style multi-GPU coarse. On PCIe-only boxes measure it first:
    # the fallback is single-GPU coarse at lower resolution (see README).
    coarse_distributed: bool = False

    coarse: CoarseConfig = field(default_factory=CoarseConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    block: BlockConfig = field(default_factory=BlockConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run_eval: bool = True

    # Redo stages even if their artifacts exist: "" (none), or one of
    # ingest/coarse/partition/finetune/merge — that stage and everything after.
    force: str = ""


_KNOWN_TYPES = {
    "TrainConfig": TrainConfig,
    "CoarseConfig": CoarseConfig,
    "PartitionConfig": PartitionConfig,
    "BlockConfig": BlockConfig,
    "MergeConfig": MergeConfig,
    "EvalConfig": EvalConfig,
    "PipelineConfig": PipelineConfig,
}
