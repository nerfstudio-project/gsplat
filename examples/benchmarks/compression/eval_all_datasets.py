# skript for running evaluations on all datasets and writing per scene csv files for 3dgs.zip survey

import os
import json
import csv
import subprocess
import shutil

RESULT_DIR_STRING="results/benchmark_{dataset}_mcmc_{num_g}_png_compression"

datasets = ["tt", "db", "mip"]

num_gaussians= [
    (360000,  "0_36M"),
    (490000,  "0_49M"),
    (1000000, "1M"),
    (4000000, "4M")
    ]

named_methods = {
    1000000: "-1.00M"
}

def run_all_evals():
    for dataset in datasets:
        for num_g in num_gaussians:

            RESULT_DIR = RESULT_DIR_STRING.format(dataset=dataset, num_g=num_g[1])
            CAP_MAX = num_g[0]

            if dataset == "mip":
                filename = "mcmc.sh"
            else:
                filename = f"mcmc_{dataset}.sh"

            print(f"Running {dataset} with {CAP_MAX} Gaussians, writing results to {RESULT_DIR}")
            os.system(f"bash benchmarks/compression/{filename} {RESULT_DIR} {CAP_MAX}")


def write_results():   
    # fetch per-scene results and write into csvs
    for dataset in datasets:
        
        if dataset == "tt":
                dataset_name = "TanksAndTemples"
        elif dataset == "db":
            dataset_name = "DeepBlending"
        elif dataset == "mip":
            dataset_name = "MipNeRF360"
        
        csv_dir = os.path.join("benchmarks", "compression", "results", dataset_name)

        if os.path.exists(csv_dir):
            shutil.rmtree(csv_dir)
        os.makedirs(csv_dir)

        for num_g in num_gaussians:
            RESULT_DIR = RESULT_DIR_STRING.format(dataset=dataset, num_g=num_g[1])
            try:
                scenes = os.listdir(RESULT_DIR)
            except:
                print("no results for", RESULT_DIR)

            for scene in scenes:
                scene_dir = os.path.join(RESULT_DIR, scene)
                csv_file_path = os.path.join(csv_dir, scene+".csv")
                metrics_path = os.path.join(RESULT_DIR, scene, "stats", "compress_step29999.json")
                if not os.path.exists(scene_dir):
                    continue

                #extract size
                zip_path = f"{scene_dir}/compression.zip"
                if os.path.exists(zip_path):
                    subprocess.run(f"rm {zip_path}", shell=True)
                subprocess.run(f"zip -r {zip_path} {scene_dir}/compression/", shell=True)
                out = subprocess.run(
                    f"stat -c%s {zip_path}", shell=True, capture_output=True
                )
                size = int(out.stdout)

                #extract metrics
                with open(metrics_path, 'r') as file:
                    data = json.load(file)
                
                if num_g[0] in named_methods:
                    name = named_methods[num_g[0]]
                else:
                    name = ""

                new_row = [name, data["psnr"], data["ssim"], data["lpips"], size, num_g[0]]

                with open(csv_file_path, 'a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    
                    # Check if the file is empty before writing the header
                    if os.path.getsize(csv_file_path) == 0:
                        header = ["Submethod", "PSNR", "SSIM", "LPIPS", "Size [Bytes]", "#Gaussians"]
                        writer.writerow(header)
                    
                    writer.writerow(new_row)

if __name__ == "__main__":
    run_all_evals()
    write_results()


        

