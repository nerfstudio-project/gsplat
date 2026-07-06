"""End-to-end orchestrator: ingest -> coarse -> partition -> finetune ->
merge -> eval.

Every stage is idempotent: it checks its output artifacts and skips
itself when they exist, so a killed pipeline resumes with the same
command. ``--force <stage>`` redoes that stage and everything after it.

Heavy stages run as subprocesses (with ``--config-json``) so the
orchestrator process never holds a CUDA context, and each stage's GPU
memory dies with its process. The finetune stage is the work queue —
one block per GPU, longest first.

Run::

    python -m citygs.pipeline --data-dir data/city --result-dir results/city \
        --gpus 0 1 2 3 4 5 6 7
"""

import os
import subprocess
import sys

import tyro

from .config import PipelineConfig, save_config
from .scene.colmap import IngestConfig, ingest
from .train.finetune_block import block_ckpt_path

STAGES = ["ingest", "coarse", "partition", "finetune", "merge", "eval"]


def _forced(cfg: PipelineConfig, stage: str) -> bool:
    if not cfg.force:
        return False
    assert cfg.force in STAGES, f"--force must be one of {STAGES}"
    return STAGES.index(stage) >= STAGES.index(cfg.force)


def _run_stage(module: str, cfg_obj, cfg_path: str, gpus, extra=None) -> None:
    save_config(cfg_obj, cfg_path)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    cmd = [sys.executable, "-m", module, "--config-json", cfg_path] + (extra or [])
    print(
        f"[pipeline] $ CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {' '.join(cmd)}"
    )
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"stage {module} failed (rc={proc.returncode})")


def run_pipeline(cfg: PipelineConfig) -> None:
    result_dir = os.path.abspath(cfg.result_dir)
    os.makedirs(result_dir, exist_ok=True)
    cfg_dir = os.path.join(result_dir, "configs")

    # Single source of truth for stage wiring: everything hangs off
    # result_dir; user-set paths in sub-configs are overridden.
    manifest_path = os.path.join(result_dir, "scene.json")
    cfg.coarse.manifest = manifest_path
    cfg.coarse.result_dir = os.path.join(result_dir, "coarse")
    cfg.partition.manifest = manifest_path
    cfg.partition.result_dir = os.path.join(result_dir, "partition")
    cfg.block.manifest = manifest_path
    cfg.block.result_dir = os.path.join(result_dir, "blocks")
    cfg.merge.manifest = manifest_path
    cfg.merge.blocks_dir = cfg.block.result_dir
    cfg.merge.result_dir = os.path.join(result_dir, "merged")
    cfg.eval.manifest = manifest_path
    cfg.eval.ckpt = os.path.join(cfg.merge.result_dir, "merged.pt")
    cfg.eval.result_dir = os.path.join(result_dir, "eval")
    save_config(cfg, os.path.join(cfg_dir, "pipeline.json"))

    # ---------------------------------------------------------- 1. ingest
    if os.path.isfile(manifest_path) and not _forced(cfg, "ingest"):
        print(f"[pipeline] ingest: {manifest_path} exists, skipping")
    else:
        ingest(
            IngestConfig(
                data_dir=cfg.data_dir,
                result_dir=result_dir,
                factor=cfg.data_factor,
                test_every=cfg.test_every,
            )
        )

    from .scene.manifest import SceneManifest

    # ---------------------------------------------------------- 2. coarse
    coarse_out = os.path.join(cfg.coarse.result_dir, "coarse.pt")
    if os.path.isfile(coarse_out) and not _forced(cfg, "coarse"):
        print(f"[pipeline] coarse: {coarse_out} exists, skipping")
    else:
        gpus = cfg.gpus if cfg.coarse_distributed else cfg.gpus[:1]
        _run_stage(
            "citygs.train.coarse",
            cfg.coarse,
            os.path.join(cfg_dir, "coarse.json"),
            gpus,
        )

    # -------------------------------------------------------- 3. partition
    manifest = SceneManifest.load(manifest_path)
    if (
        manifest.blocks
        and "gaussian_block_ids" in manifest.artifacts
        and not _forced(cfg, "partition")
    ):
        print(f"[pipeline] partition: {len(manifest.blocks)} blocks exist, skipping")
    else:
        _run_stage(
            "citygs.partition.run",
            cfg.partition,
            os.path.join(cfg_dir, "partition.json"),
            cfg.gpus[:1],
        )
        manifest = SceneManifest.load(manifest_path)

    # --------------------------------------------------------- 4. finetune
    missing = [
        b.block_id
        for b in manifest.blocks
        if not os.path.isfile(block_ckpt_path(cfg.block.result_dir, b.block_id))
    ]
    if not missing and not _forced(cfg, "finetune"):
        print("[pipeline] finetune: all block checkpoints exist, skipping")
    else:
        from .train.scheduler import schedule_blocks

        result = schedule_blocks(cfg.block, cfg.gpus, force=_forced(cfg, "finetune"))
        if result.failed:
            raise RuntimeError(
                f"finetune failed for blocks {sorted(result.failed)}; "
                "fix and rerun the pipeline (finished blocks are skipped)"
            )

    # ------------------------------------------------------------ 5. merge
    merged_out = os.path.join(cfg.merge.result_dir, "merged.pt")
    if os.path.isfile(merged_out) and not _forced(cfg, "merge"):
        print(f"[pipeline] merge: {merged_out} exists, skipping")
    else:
        _run_stage(
            "citygs.merge",
            cfg.merge,
            os.path.join(cfg_dir, "merge.json"),
            cfg.gpus[:1],
        )

    # ------------------------------------------------------------- 6. eval
    if cfg.run_eval:
        metrics = os.path.join(cfg.eval.result_dir, "metrics.json")
        if os.path.isfile(metrics) and not _forced(cfg, "eval"):
            print(f"[pipeline] eval: {metrics} exists, skipping")
        else:
            _run_stage(
                "citygs.eval",
                cfg.eval,
                os.path.join(cfg_dir, "eval.json"),
                cfg.gpus[:1],
            )

    print(f"[pipeline] done. Inspect with:")
    print(
        f"  python -m citygs.viz.viewer --manifest {manifest_path} "
        f"--blocks-dir {cfg.block.result_dir}"
    )


def main() -> None:
    run_pipeline(tyro.cli(PipelineConfig))


if __name__ == "__main__":
    main()
