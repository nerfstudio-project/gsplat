import torch
import numpy as np
import imageio
import os
from gsplat.compression import compress_splats, decompress_splats
import json


def main():
    device = "cuda:0"
    ckpt_path = "examples/results/360_v2/3dgs/garden/ckpts/ckpt_29999.pt"
    compress_dir = "examples/results/test_compress"
    if not os.path.exists(compress_dir):
        os.makedirs(compress_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device)
    splats0 = ckpt["splats"]
    # torch.save({"splats": ckpt["splats"]}, os.path.join(out_dir, "splats.pt"))
    compress_splats(compress_dir, splats0)
    splats1 = decompress_splats(compress_dir)
    for k in splats1.keys():
        attr0 = splats0[k]
        attr1 = splats1[k].to(attr0.device)
        print(
            k,
            attr0.shape,
            attr1.shape,
            attr0.dtype,
            attr1.dtype,
        )
        if attr0.numel() != 0:
            print((attr0 - attr1).abs().max())


def gif():
    # results_dir = "examples/results/360_v2/3dgs_sort"
    results_dir = "examples/results/360_v2/3dgs_sort"
    scenes = ["garden", "bicycle", "stump", "bonsai", "counter", "kitchen", "room"]
    psnrs = []
    for scene in scenes:
        scene_dir = os.path.join(results_dir, scene)
        # ckpt_dir = os.path.join(scene_dir, "ckpts")
        stats_path = os.path.join(scene_dir, f"stats/val_step29999.json")
        # stats_path = os.path.join(scene_dir, f"compressed_results/stats/val_step29999.json")

        with open(stats_path, "r") as f:
            stats = json.load(f)
            psnrs.append(stats["psnr"])
    print(psnrs)
    print(np.mean(psnrs))

    # writer = imageio.get_writer(f"{ckpt_dir}/{scene}_mcmc_grid.mp4", fps=10)

    # for step in range(4500, 5500, 100):
    #     grid = imageio.imread(os.path.join(ckpt_dir, f"grid_step{step:04d}.png"))
    #     writer.append_data(grid)
    # writer.close()


if __name__ == "__main__":
    main()
    # gif()
