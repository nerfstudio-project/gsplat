# Training script for the Mip-NeRF 360 dataset
# The model is trained with downsampling factor 8 and rendered with downsampling factor 1, 2, 4, 8

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import glob

# 9 scenes
# scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]
# factors = [4, 2, 2, 4, 4, 4, 4, 2, 2]

# 7 scenes
scenes = ["bicycle", "bonsai", "counter", "garden", "stump", "kitchen", "room"]
factors = [8] * len(scenes)

excluded_gpus = set([])

# classic
result_dir = "results/benchmark_stmt"
# antialiased
result_dir = "results/benchmark_antialiased_stmt"
# mip-splatting
# result_dir = "results/benchmark_mipsplatting_stmt"

dry_run = False

jobs = list(zip(scenes, factors))


def train_scene(gpu, scene, factor):
    # train without eval
    # classic
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python simple_trainer.py --eval_steps -1 --disable_viewer --data_factor {factor} --data_dir data/360_v2/{scene} --result_dir {result_dir}/{scene}"
    
    # anti-aliased
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python simple_trainer.py --eval_steps -1 --disable_viewer --data_factor {factor} --data_dir data/360_v2/{scene} --result_dir {result_dir}/{scene} --antialiased"
    
    # mip-splatting
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python simple_trainer_mip_splatting.py --eval_steps -1 --disable_viewer --data_factor {factor} --data_dir data/360_v2/{scene} --result_dir {result_dir}/{scene} --antialiased --kernel_size 0.1"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    # eval and render for all the ckpts
    ckpts = glob.glob(f"{result_dir}/{scene}/ckpts/*.pt")
    for ckpt in ckpts:
        for test_factor in [1, 2, 4, 8]:
            # classic
            # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python simple_trainer.py --disable_viewer --data_factor {test_factor} --data_dir data/360_v2/{scene} --result_dir {result_dir}/{scene}_{test_factor} --ckpt {ckpt}"
            
            # anti-aliased
            # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python simple_trainer.py --disable_viewer --data_factor {test_factor} --data_dir data/360_v2/{scene} --result_dir {result_dir}/{scene}_{test_factor} --ckpt {ckpt} --antialiased"
            
            # mip-splatting
            cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python simple_trainer_mip_splatting.py --disable_viewer --data_factor {test_factor} --data_dir data/360_v2/{scene} --result_dir {result_dir}/{scene}_{test_factor} --ckpt {ckpt} --antialiased --kernel_size 0.1"
            print(cmd)
            if not dry_run:
                os.system(cmd)

    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(
            GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1, maxLoad=0.1)
        )
        # all_available_gpus = set([0,1,2,3])
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(
                worker, gpu, *job
            )  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing
            time.sleep(2)

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(
                future
            )  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)
