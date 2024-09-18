"""Script to download benchmark dataset(s)"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

# dataset names
dataset_names = Literal[
    "mipnerf360",
    "mipnerf360_extra",
    "bilarf_data",
    "alameda",
    "alameda_undistorted",
]

# dataset urls
urls = {
    "mipnerf360": "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
    "mipnerf360_extra": "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip",
    "bilarf_data": "https://huggingface.co/datasets/Yuehao/bilarf_data/resolve/main/bilarf_data.zip",
    "alameda": "https://storage.googleapis.com/gresearch/refraw360/zipnerf/alameda.zip",
    "alameda_undistorted": "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/alameda.zip",
}

# rename maps
dataset_rename_map = {
    "mipnerf360": "360_v2",
    "mipnerf360_extra": "360_v2",
    "bilarf_data": "bilarf",
    "alameda": "zipnerf/alameda",
    "alameda_undistorted": "zipnerf_undistorted/alameda",
}


@dataclass
class DownloadData:
    dataset: dataset_names = "mipnerf360"
    save_dir: Path = Path(os.getcwd() + "/data")

    def main(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_download(self.dataset)

    def dataset_download(self, dataset: dataset_names):
        (self.save_dir / dataset_rename_map[dataset]).mkdir(parents=True, exist_ok=True)

        file_name = Path(urls[dataset]).name

        # download
        download_command = [
            "curl",
            "-o",
            str(self.save_dir / dataset_rename_map[dataset] / file_name),
            urls[dataset],
        ]
        try:
            subprocess.run(download_command, check=True)
            print("File file downloaded succesfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading file: {e}")

        # if .zip
        if Path(urls[dataset]).suffix == ".zip":
            if os.name == "nt":  # Windows doesn't have 'unzip' but 'tar' works
                extract_command = [
                    "tar",
                    "-xvf",
                    self.save_dir / dataset_rename_map[dataset] / file_name,
                    "-C",
                    self.save_dir / dataset_rename_map[dataset],
                ]
            else:
                extract_command = [
                    "unzip",
                    self.save_dir / dataset_rename_map[dataset] / file_name,
                    "-d",
                    self.save_dir / dataset_rename_map[dataset],
                ]
        # if .tar
        else:
            extract_command = [
                "tar",
                "-xvzf",
                self.save_dir / dataset_rename_map[dataset] / file_name,
                "-C",
                self.save_dir / dataset_rename_map[dataset],
            ]

        # extract
        try:
            subprocess.run(extract_command, check=True)
            os.remove(self.save_dir / dataset_rename_map[dataset] / file_name)
            print("Extraction complete.")
        except subprocess.CalledProcessError as e:
            print(f"Extraction failed: {e}")


if __name__ == "__main__":
    tyro.cli(DownloadData).main()
