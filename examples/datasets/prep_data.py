from pathlib import Path
import json
import argparse
import configparser
import os


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process some data paths.")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    json_path = dataset_path / "split.json"
    clutters = []
    extras = []
    cleans = []
    with open(json_path) as f:
        split = json.load(f)
    if "extra" in split.keys():
        extras = split["extra"]
    if "clutter" in split.keys():
        clutters = split["clutter"]
    if "clean" in split.keys():
        cleans = split["clean"]

    transforms = None
    transforms_path = dataset_path / "transforms.json"
    if os.path.exists(transforms_path):
        with open(transforms_path) as f:
            transforms = json.load(f)

    image_dirs = dataset_path / "images"
    dirs = os.listdir(image_dirs)
    dirs.sort()
    root = str(image_dirs) + "/"
    file_type = dirs[0].split(".")[-1]
    for i in range(len(dirs)):
        if i in clutters:
            os.rename(
                Path(root + dirs[i]),
                Path(root + "2clutter" + str(i).zfill(3) + "." + file_type),
            )
            if transforms is not None:
                transforms["frames"][i]["file_path"] = (
                    "./images/2clutter" + str(i).zfill(3) + "." + file_type
                )
        elif i in extras:
            os.rename(
                Path(root + dirs[i]),
                Path(root + "1extra" + str(i).zfill(3) + "." + file_type),
            )
            if transforms is not None:
                transforms["frames"][i]["file_path"] = (
                    "./images/1extra" + str(i).zfill(3) + "." + file_type
                )
        elif i in cleans:
            os.rename(
                Path(root + dirs[i]),
                Path(root + "0clean" + str(i).zfill(3) + "." + file_type),
            )
            if transforms is not None:
                transforms["frames"][i]["file_path"] = (
                    "./images/0clean" + str(i).zfill(3) + "." + file_type
                )
        else:
            os.rename(
                Path(root + dirs[i]),
                Path(root + "2clutter" + str(i).zfill(3) + "." + file_type),
            )
            # if transforms is not None:
            #  transforms['frames'][i]['file_path'] = './images/2clutter' + str(i).zfill(3) + '.' + file_type
        with open(transforms_path, "w") as f:
            json.dump(transforms, f)


if __name__ == "__main__":
    main()
