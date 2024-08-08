import json
import numpy as np
import glob

# 9 scenes
# scenes = ['bicycle', 'flowers', 'garden', 'stump', 'treehill', 'room', 'counter', 'kitchen', 'bonsai']

# outdoor scenes
# scenes = scenes[:5]
# indoor scenes
# scenes = scenes[5:]

# 7 scenes
scenes = ["bicycle", "bonsai", "counter", "garden", "stump", "kitchen", "room"]

result_dirs = ["results/benchmark"]
result_dirs = ["results/benchmark_antialiased"]
result_dirs = ["results/benchmark_mipsplatting"]
result_dirs = ["results/benchmark_mipsplatting_cuda3D"]

all_metrics = {"psnr": [], "ssim": [], "lpips": [], "num_GS": []}
print(result_dirs)

for scene in scenes:
    print(scene, end=" ")
    for result_dir in result_dirs:
        json_files = glob.glob(f"{result_dir}/{scene}/stats/val_step29999.json")
        for json_file in json_files:
            # print(json_file)
            data = json.load(open(json_file))
            # print(data)

            for k in ["psnr", "ssim", "lpips", "num_GS"]:
                all_metrics[k].append(data[k])
                print(f"{data[k]:.3f}", end=" ")
            print()

latex = []
for k in ["psnr", "ssim", "lpips", "num_GS"]:
    numbers = np.asarray(all_metrics[k]).mean(axis=0).tolist()
    print(numbers)
    numbers = [numbers]
    if k == "PSNR":
        numbers = [f"{x:.2f}" for x in numbers]
    elif k == "num_GS":
        num = numbers[0] / 1e6
        numbers = [f"{num:.2f}"]
    else:
        numbers = [f"{x:.3f}" for x in numbers]
    latex.extend(numbers)
print(" | ".join(latex))
