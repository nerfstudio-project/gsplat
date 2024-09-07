"""
Script to download benchmark dataset(s)

By default, this script downloads the 'mipnerf360' dataset. 
You can specify a different dataset to download using the --dataset option. 
If you want to download all available datasets, you can set the --dataset option to 'all'.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, List
import tyro

# dataset names
dataset_names = Literal["mipnerf360", "mipnerf360_extra", "tandt", "deepblending", "all"]

# dataset urls
urls = {
    "mipnerf360": "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
    "mipnerf360_extra": "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip",
    "tandt_db": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
}

# rename maps
dataset_rename_map = {
    "mipnerf360": "360_v2",
    "mipnerf360_extra": "360_v2",
    "tandt": "tandt",
    "deepblending": "db"
}

@dataclass
class DownloadData:
    dataset: dataset_names = "mipnerf360"
    save_dir: Path = Path(os.getcwd() + "/data")

    def main(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.dataset == "all":
            for dataset in urls.keys():
                self.dataset_download(dataset)
        else:
            self.dataset_download(self.dataset)

    def dataset_download(self, dataset: str):
        if dataset in ["tandt", "deepblending", "tandt_db"]:
            url = urls["tandt_db"]
            file_name = Path(url).name
            extract_dir = self.save_dir
        else:
            url = urls[dataset]
            file_name = Path(url).name
            extract_dir = self.save_dir / dataset_rename_map[dataset]

        # download
        download_command = [
            "wget",
            "-P",
            str(extract_dir),
            url,
        ]
        try:
            subprocess.run(download_command, check=True)
            print(f"File {file_name} downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading file: {e}")
            return

        # if .zip
        if Path(url).suffix == ".zip":
            extract_command = [
                "unzip",
                "-o",
                extract_dir / file_name,
                "-d",
                extract_dir,
            ]
        # if .tar
        else:
            extract_command = [
                "tar",
                "-xvzf",
                extract_dir / file_name,
                "-C",
                extract_dir,
            ]

        try:
            subprocess.run(extract_command, check=True)
            os.remove(extract_dir / file_name)
            print("Extraction complete.")
        except subprocess.CalledProcessError as e:
            print(f"Extraction failed: {e}")

        # For tandt_db, we need to rename the extracted folders
        if dataset in ["tandt", "deepblending"]:
            os.rename(self.save_dir / "tandt", self.save_dir / dataset_rename_map["tandt"])
            os.rename(self.save_dir / "db", self.save_dir / dataset_rename_map["deepblending"])

if __name__ == "__main__":
    tyro.cli(DownloadData).main()