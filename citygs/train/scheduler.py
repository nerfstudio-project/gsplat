"""Work-queue scheduler: N blocks onto M GPUs, one block per GPU process.

Blocks are queued longest-first (LPT), which combined with the dynamic
queue gives near-optimal makespan without any explicit bin packing. Each
block runs as a subprocess pinned via CUDA_VISIBLE_DEVICES, logs to its
own file, and drops a checkpoint that doubles as its done marker — so a
killed run resumes by simply skipping finished blocks.

Run: ``python -m citygs.train.scheduler --manifest ... --gpus 0 1 2 3``
"""

import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from ..config import BlockConfig, load_config, save_config
from ..scene.manifest import SceneManifest
from .finetune_block import block_ckpt_path


@dataclass
class Job:
    block_id: int
    cost: float
    cmd: List[str]
    log_path: str


@dataclass
class QueueResult:
    succeeded: List[int] = field(default_factory=list)
    failed: List[int] = field(default_factory=list)
    skipped: List[int] = field(default_factory=list)


def run_queue(
    jobs: Sequence[Job],
    gpus: Sequence[int],
    max_retries: int = 1,
    env_for_gpu: Optional[Callable[[int], dict]] = None,
) -> QueueResult:
    """Run jobs longest-first on one worker thread per GPU."""
    todo: "queue.PriorityQueue" = queue.PriorityQueue()
    for order, job in enumerate(sorted(jobs, key=lambda j: -j.cost)):
        todo.put((order, job))

    result = QueueResult()
    lock = threading.Lock()

    def worker(gpu: int):
        while True:
            try:
                _, job = todo.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            if env_for_gpu is not None:
                env.update(env_for_gpu(gpu))
            ok = False
            for attempt in range(max_retries + 1):
                tic = time.time()
                os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
                with open(job.log_path, "a") as log:
                    log.write(
                        f"\n=== block {job.block_id} gpu {gpu} attempt {attempt} ===\n"
                    )
                    log.flush()
                    proc = subprocess.run(job.cmd, env=env, stdout=log, stderr=log)
                dt = time.time() - tic
                if proc.returncode == 0:
                    print(
                        f"[queue] block {job.block_id} done on gpu {gpu} "
                        f"({dt / 60:.1f} min)",
                        flush=True,
                    )
                    ok = True
                    break
                print(
                    f"[queue] block {job.block_id} FAILED on gpu {gpu} "
                    f"(attempt {attempt}, rc={proc.returncode}) — see {job.log_path}",
                    flush=True,
                )
            with lock:
                (result.succeeded if ok else result.failed).append(job.block_id)
            todo.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return result


def schedule_blocks(
    block_cfg: BlockConfig, gpus: Sequence[int], force: bool = False
) -> QueueResult:
    """Build the job list from the manifest and run it."""
    manifest = SceneManifest.load(block_cfg.manifest)
    assert manifest.blocks, "no blocks in manifest — run the partition stage first"
    os.makedirs(block_cfg.result_dir, exist_ok=True)

    # One shared config file; the per-block id is passed on the command line.
    cfg_json = os.path.join(block_cfg.result_dir, "block_config.json")
    save_config(block_cfg, cfg_json)

    jobs, result0 = [], QueueResult()
    for block in manifest.blocks:
        out = block_ckpt_path(block_cfg.result_dir, block.block_id)
        if os.path.isfile(out) and not force:
            result0.skipped.append(block.block_id)
            continue
        jobs.append(
            Job(
                block_id=block.block_id,
                cost=block.cost,
                cmd=[
                    sys.executable,
                    "-m",
                    "citygs.train.finetune_block",
                    "--config-json",
                    cfg_json,
                    "--block-id",
                    str(block.block_id),
                ],
                log_path=os.path.join(
                    block_cfg.result_dir, f"block_{block.block_id:04d}", "log.txt"
                ),
            )
        )

    if result0.skipped:
        print(f"[queue] skipping finished blocks: {sorted(result0.skipped)}")
    print(f"[queue] {len(jobs)} blocks on GPUs {list(gpus)} (longest first)")
    result = run_queue(jobs, gpus)
    result.skipped = result0.skipped
    if result.failed:
        print(f"[queue] FAILED blocks: {sorted(result.failed)} — rerun to retry")
    return result


def main(argv=None) -> None:
    import tyro
    from dataclasses import dataclass as _dc

    @_dc
    class Args:
        config_json: str = ""
        manifest: str = "results/scene.json"
        result_dir: str = "results/blocks"
        gpus: List[int] = field(default_factory=lambda: [0])
        force: bool = False

    args = tyro.cli(Args, args=argv)
    if args.config_json:
        block_cfg = load_config(BlockConfig, args.config_json)
    else:
        block_cfg = BlockConfig(manifest=args.manifest, result_dir=args.result_dir)
    result = schedule_blocks(block_cfg, args.gpus, force=args.force)
    sys.exit(1 if result.failed else 0)


if __name__ == "__main__":
    main()
