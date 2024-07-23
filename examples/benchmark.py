# Benchmark script

import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import GPUtil


@dataclass
class BenchmarkConfig:
    """Benchmark config"""

    # trainer to run
    trainer: str = "simple_trainer.py"
    # path to data
    data_dir: str = "data/360_v2"
    # scenes to run
    scenes: set = (
        "bicycle",
        "bonsai",
        "counter",
        "garden",
        "stump",
        "kitchen",
        "room",
    )
    # downscale factors
    factors: set = (4, 2, 2, 4, 4, 2, 2)
    # exclude gpus
    excluded_gpus: set = field(default_factory=set)
    # result directory
    result_dir: str = "results/baseline"
    # dry run, useful for debugging
    dry_run: bool = False
    # extra model specific configs
    model_configs: dict = field(default_factory=dict)


# Configurations to run
baseline_config = BenchmarkConfig()
absgrad_config = BenchmarkConfig(
    result_dir="results/absgrad",
    model_configs={"--absgrad": True, "--grow_grad2d": 0.0006},
)
antialiased_config = BenchmarkConfig(
    result_dir="results/antialiased", model_configs={"--antialiased": True}
)
mcmc_config = BenchmarkConfig(
    trainer="simple_trainer_mcmc.py",
    result_dir="results/mcmc",
)

configs_to_run = [
    baseline_config,
    # mcmc_config,
    # absgrad_config,
    # antialiased_config,
]


def train_scene(gpu, scene, factor, config):
    """Train a single scene with config on current gpu"""
    # additional user set model configs
    model_config_args = " ".join(f"{k} {v}" for k, v in config.model_configs.items())

    # train without eval
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python {config.trainer} --eval_steps -1 --disable_viewer --data_factor {factor} --data_dir {config.data_dir}/{scene} --result_dir {config.result_dir}/{scene} {model_config_args}"

    print(cmd)
    if not config.dry_run:
        os.system(cmd)

    # eval and render for all the ckpts
    ckpts = glob.glob(f"{config.result_dir}/{scene}/ckpts/*.pt")
    for ckpt in ckpts:
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python {config.trainer} --disable_viewer --data_factor {factor} --data_dir {config.data_dir}//{scene} --result_dir {config.result_dir}/{scene} --ckpt {ckpt} {model_config_args}"
        print(cmd)
        if not config.dry_run:
            os.system(cmd)

    return True


def worker(gpu, scene, factor, config):
    """This worker function starts a job and returns when it's done."""
    print(f"Starting {config.trainer} job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor, config)
    print(f"Finished {config.trainer} job on GPU {gpu} with scene {scene}\n")


def dispatch_jobs(jobs, executor, config):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(
            GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1, maxLoad=0.1)
        )
        available_gpus = list(all_available_gpus - reserved_gpus - config.excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(
                worker, gpu, *job, config
            )  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)
            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(
                future
            )  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., releasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


def main():
    """Launch batch_configs in serial but process each config in parallel (multi gpu)"""

    for config in configs_to_run:
        # num jobs = num scenes to run for current config
        jobs = list(zip(config.scenes, config.factors))

        # Run multiple gpu train scripts
        # Using ThreadPoolExecutor to manage the thread pool
        with ThreadPoolExecutor(max_workers=8) as executor:
            dispatch_jobs(jobs, executor, config)


if __name__ == "__main__":
    main()
