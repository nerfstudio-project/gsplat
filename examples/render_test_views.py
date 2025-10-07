import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import imageio.v3 as iio
import torch
import yaml
from datasets.colmap import Dataset, Parser
from datasets.colmap_rgba import DatasetRGBA
from easydict import EasyDict as edict
from simple_trainer_rgba import Runner


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Render test views for all experiments and frames"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="Starting frame number (default: 0)"
    )
    parser.add_argument(
        "--end-frame", type=int, default=49, help="Ending frame number (default: 49)"
    )
    return parser.parse_args()


def get_all_experiments() -> dict[str, list[str]]:
    """Get all available experiments."""
    return {
        "rgba": [
            "mcmc_random_bkgd",
            "mcmc_alpha_loss",
            "default_random_bkgd",
            "default_alpha_loss",
        ],
        "rgb": ["default", "mcmc"],
    }


def load_config(result_dir: str) -> edict:
    """Load configuration from result directory."""
    config_path = os.path.join(result_dir, "cfg.yml")
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    return edict(cfg)


def create_test_dataloader(
    test_data_dir: str, data_type: str, cfg: edict
) -> torch.utils.data.DataLoader:
    """Create test data loader."""
    test_parser = Parser(
        data_dir=test_data_dir,
        factor=cfg.data_factor,
        normalize=cfg.normalize_world_space,
        test_every=1,
    )

    if data_type == "rgba":
        test_dataset = DatasetRGBA(test_parser, split="eval")
    else:
        test_dataset = Dataset(test_parser, split="eval")

    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=7,
        shuffle=False,
        num_workers=0,
    )


def setup_runner_and_load_checkpoint(
    cfg: edict, result_dir: str, device: torch.device
) -> Runner:
    """Setup runner and load checkpoint."""
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)

    ckpt_path = os.path.join(result_dir, "ckpts", "ckpt_29999_rank0.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    runner.splats = ckpt["splats"]

    return runner


def render_test_views(
    runner: Runner, data: dict, cfg: edict, device: torch.device
) -> torch.Tensor:
    """Render test views using the runner."""
    camtoworlds = data["camtoworld"].to(device)
    Ks = data["K"].to(device)
    pixels = data["image"].to(device) / 255.0

    # Check if image is RGB (3 channels) or RGBA (4 channels)
    if pixels.shape[-1] == 4:
        # RGBA case
        pixels_alpha = pixels[..., 3:]
        pixels = pixels[..., :3] * pixels_alpha  # Alpha blend for RGBA
    else:
        # RGB case - no alpha blending needed
        pixels = pixels[..., :3]

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

    # Concatenate renders and alphas to get RGBA only if original had alpha
    if pixels.shape[-1] == 4:
        renders = torch.cat([colors, alphas], dim=-1)
    else:
        renders = colors
    return renders


def save_rendered_images(
    renders: torch.Tensor,
    test_data_dir: str,
    experiment: str,
    frame: str,
    camera_ids: list,
) -> None:
    """Save rendered images to disk."""
    renders_np = renders.cpu().detach().numpy()
    renders_np = (renders_np * 255.0).astype("uint8")

    save_dir = os.path.join(test_data_dir, "gsplat_renders", experiment)
    os.makedirs(save_dir, exist_ok=True)

    for i, camera_id in enumerate(camera_ids):
        print(f"Saving rendered image for camera {camera_id}...")
        img_name = f"{frame}_{camera_id:03d}.png"
        render = renders_np[i, ...]
        iio.imwrite(os.path.join(save_dir, img_name), render)


def process_single_experiment(task_info: tuple) -> tuple[str, bool]:
    """Process a single experiment for a given frame. Returns (run_info, success)."""
    data_type, experiment, frame_str = task_info

    # Set up device for this process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        print(f"Processing {data_type}/{experiment} for frame {frame_str}...")

        # Setup paths and configuration
        result_dir = f"/home/minhtran/Code/data/vocap/minh_2/gsplat_results/frames/{frame_str}/{data_type}/{experiment}"
        test_data_dir = (
            f"/home/minhtran/Code/data/vocap/minh_2/frames/{frame_str}/test/{data_type}"
        )

        # Check if result directory exists
        if not os.path.exists(result_dir):
            raise FileNotFoundError(f"Result directory not found: {result_dir}")

        # Check if test data directory exists
        if not os.path.exists(test_data_dir):
            raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")

        cfg = load_config(result_dir)

        # Create test dataloader
        test_loader = create_test_dataloader(test_data_dir, data_type, cfg)

        # Setup runner and load checkpoint
        runner = setup_runner_and_load_checkpoint(cfg, result_dir, device)

        # Render test views
        data = next(iter(test_loader))
        renders_rgba = render_test_views(runner, data, cfg, device)

        # Save rendered images
        test_parser = Parser(
            data_dir=test_data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=1,
        )
        save_rendered_images(
            renders_rgba, test_data_dir, experiment, frame_str, test_parser.camera_ids
        )

        run_info = f"{data_type}/{experiment}/frame_{frame_str}"
        print(f"Successfully processed {run_info}")
        return (run_info, True)

    except Exception as e:
        run_info = f"{data_type}/{experiment}/frame_{frame_str}"
        print(f"ERROR processing {run_info}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return (run_info, False)


def main():
    """Main function to render test views for all experiments and frames."""
    # Parse arguments
    args = parse_arguments()

    # Get all experiments
    all_experiments = get_all_experiments()

    # Create list of all tasks
    tasks = []
    for frame_num in range(args.start_frame, args.end_frame + 1):
        frame_str = f"{frame_num:08d}"
        for data_type, experiments in all_experiments.items():
            for experiment in experiments:
                tasks.append((data_type, experiment, frame_str))

    # Track results
    successful_runs = []
    failed_runs = []

    # Process tasks in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_experiment, task): task for task in tasks
        }

        # Process completed tasks
        for future in as_completed(future_to_task):
            run_info, success = future.result()

            if success:
                successful_runs.append(run_info)
            else:
                failed_runs.append(run_info)

    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total successful runs: {len(successful_runs)}")
    print(f"Total failed runs: {len(failed_runs)}")

    if failed_runs:
        print("\nFAILED RUNS:")
        for failed_run in failed_runs:
            print(f"  - {failed_run}")

        # Exit with error code if there were failures
        print(f"\nERROR: {len(failed_runs)} runs failed. See details above.")
        exit(1)
    else:
        print("\nAll runs completed successfully!")


if __name__ == "__main__":
    main()


# TODO: Check these failed experiments

# FAILED RUNS:
#   - rgb/default/frame_00000000
#   - rgb/mcmc/frame_00000000
#   - rgb/default/frame_00000003
#   - rgb/mcmc/frame_00000003
#   - rgba/mcmc_random_bkgd/frame_00000008
#   - rgba/mcmc_alpha_loss/frame_00000008
#   - rgba/default_random_bkgd/frame_00000008
#   - rgba/default_alpha_loss/frame_00000008
#   - rgb/default/frame_00000008
#   - rgb/mcmc/frame_00000008
#   - rgba/mcmc_random_bkgd/frame_00000009
#   - rgba/mcmc_alpha_loss/frame_00000009
#   - rgba/default_random_bkgd/frame_00000009
#   - rgba/default_alpha_loss/frame_00000009
#   - rgb/default/frame_00000009
#   - rgb/mcmc/frame_00000009
#   - rgb/default/frame_00000014
#   - rgb/mcmc/frame_00000014
#   - rgb/default/frame_00000034
#   - rgb/mcmc/frame_00000034
