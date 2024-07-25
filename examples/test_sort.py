# import torch
# import numpy as np
# import imageio
# import os
# from gsplat.compression import compress_splats, decompress_splats
# import json
# import subprocess
# from collections import defaultdict


# from plas import sort_with_plas
# def sort_params(params_to_sort):
#     cap_max = params_to_sort.shape[0]

#     shuffled_indices = torch.randperm(
#         params_to_sort.shape[0], device=params_to_sort.device
#     )
#     params_to_sort = params_to_sort[shuffled_indices]
#     n_sidelen = int(cap_max**0.5)
#     grid = params_to_sort.reshape((n_sidelen, n_sidelen, -1))
#     _, sorted_indices = sort_with_plas(
#         grid.permute(2, 0, 1), improvement_break=1e-4, verbose=True
#     )
#     sorted_indices = sorted_indices.squeeze().flatten()
#     sorted_indices = shuffled_indices[sorted_indices]
#     return sorted_indices


# def main():
#     results_dir = "examples/results/360_v2/3dgs+sq"
#     out_dir = "examples/results/test_sort/"
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir, exist_ok=True)

#     scenes = ["garden"] #, "bicycle", "stump", "bonsai", "counter", "kitchen", "room"]

#     for scene in scenes:
#         scene_dir = os.path.join(results_dir, scene)
#         keys = ["means_u", "scales_u", "quats", "opacities", "sh0"]

#         arrs = []
#         arr_shapes = []
#         for key in keys:
#             arr = imageio.imread(os.path.join(scene_dir, f"compress/{key}.png"))
#             arr_shapes.append(arr.shape)

#             if arr.ndim == 2:
#                 arr = arr[:, :, None]
#             arrs.append(arr)
#         arrs = np.concatenate(arrs, axis=-1)
#         arrs = arrs / 255.0

#         grid = torch.tensor(arrs, dtype=torch.float32).cuda()
#         params = grid.reshape((grid.shape[0] * grid.shape[1], -1))

#         sort_indicies = sort_params(params)
#         params = params[sort_indicies]
#         grid = params.reshape((grid.shape[0], grid.shape[1], -1))
#         print(grid.shape)

#         grid = grid.detach().cpu().numpy()
#         grid = (grid * 255).astype(np.uint8)
#         keys = ["means_u", "means_l", "scales_u", "scales_l", "quats", "opacities", "sh0"]
#         imageio.imwrite(os.path.join(out_dir, "means_u.png"), grid[..., :3])
#         imageio.imwrite(os.path.join(out_dir, "scales_u.png"), grid[..., 3:6])
#         imageio.imwrite(os.path.join(out_dir, "quats.png"), grid[..., 6:10])
#         imageio.imwrite(os.path.join(out_dir, "opacities.png"), grid[..., 10:11][..., 0])
#         imageio.imwrite(os.path.join(out_dir, "sh0.png"), grid[..., 11:14])
#         # imageio.imwrite(os.path.join(out_dir, "means_l.png"), grid[..., 14:17])
#         # imageio.imwrite(os.path.join(out_dir, "scales_l.png"), grid[..., 17:20])


# if __name__ == "__main__":
#     main()
