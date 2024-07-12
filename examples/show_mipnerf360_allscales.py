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

result_dirs = ["results/benchmark_stmt"]
# result_dirs = ["results/benchmark_antialiased_stmt"]
# result_dirs = ["results/benchmark_mipsplatting_stmt"]

all_metrics = {"psnr": [], "ssim": [], "lpips": [], "num_GS": []}
print(result_dirs)

for scene in scenes:
    print(scene)
    for result_dir in result_dirs:
        for scale in ["8", "4", "2", "1"]:
            json_files = glob.glob(f"{result_dir}/{scene}_{scale}/stats/val_step29999.json")
            for json_file in json_files:    
                data = json.load(open(json_file))
                for k in ["psnr", "ssim", "lpips", "num_GS"]:
                    all_metrics[k].append(data[k])
                    print(f"{data[k]:.3f}", end=" ")
                print()

latex = []
for k in ["psnr", "ssim", "lpips"]:
    numbers = np.asarray(all_metrics[k]).reshape(-1, 4).mean(axis=0).tolist()
    numbers = numbers + [np.mean(numbers)]
    print(numbers)
    if k == "psnr":
        numbers = [f"{x:.2f}" for x in numbers]
    else:
        numbers = [f"{x:.3f}" for x in numbers]
    latex.extend(numbers)
print(" | ".join(latex))
