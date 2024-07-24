import torch
import numpy as np
import imageio
import os
from gsplat.compression import compress_splats, decompress_splats


def main():
    device = "cuda:0"
    ckpt_path = "examples/results/360_v2/3dgs_sort/garden/ckpts/ckpt_14999.pt"
    compress_dir = "examples/results/compress"
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

    # splats = ckpt["splats"]
    # n_gs = len(splats["means"])
    # n_sidelen = int(n_gs**0.5)
    # params = torch.cat([splats[k].reshape(n_gs, -1) for k in splats.keys()], dim=-1)
    # grid = params.reshape((n_sidelen, n_sidelen, -1))
    # grid_rgb = sh_to_rgb(grid[:, :, -3:])
    # grid_rgb = torch.clamp(grid_rgb, 0.0, 1.0)
    # grid_rgb = grid_rgb.detach().cpu().numpy()
    # grid_rgb = (grid_rgb * 255).astype(np.uint8)
    # imageio.imwrite(os.path.join(out_dir, "rgb.png"), grid_rgb)
    # imageio.imwrite(os.path.join(out_dir, "rgb.jpg"), grid_rgb)
    # imagecodecs.imwrite(os.path.join(out_dir, "rgb.jxl"), grid_rgb)


# def gif():
#     scenes = ["garden", "bicycle", "stump", "treehill", "flowers", "bonsai"]
#     for scene in tqdm(scenes):
#         ckpt_dir = f"examples/results/360_v2/3dgs_sh0_sort/{scene}/ckpts"

#         writer = imageio.get_writer(f"{ckpt_dir}/{scene}_mcmc_grid.mp4", fps=10)

#         for step in range(500, 30000, 100):
#             grid = imageio.imread(os.path.join(ckpt_dir, f"grid_step{step:04d}.png"))
#             if grid.shape[0] != 0:
#                 img = np.zeros((1000, 1000, 3), dtype=np.uint8)
#                 img[: grid.shape[0], : grid.shape[1], :] = grid
#             else:
#                 img = grid

#             writer.append_data(img)
#         writer.close()


if __name__ == "__main__":
    main()
    # gif()
