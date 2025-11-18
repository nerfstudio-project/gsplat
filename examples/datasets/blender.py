from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Literal

import imageio.v2 as imageio
import numpy as np
import torch


@dataclass
class Dataset:
    """A simple dataset class for synthetic blender data."""

    data_dir: str
    """The path to the blender scene, consisting of renders and transforms.json"""
    split: Literal["train", "test", "val"] = "train"
    """Which split to use."""

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        transforms_path = self.data_dir / f"transforms_{self.split}.json"
        with transforms_path.open("r") as transforms_handle:
            transforms = json.load(transforms_handle)
        image_ids = []
        cam_to_worlds = []
        images = []
        for frame in transforms["frames"]:
            image_id = frame["file_path"].replace("./", "")
            image_ids.append(image_id)
            file_path = self.data_dir / f"{image_id}.png"
            images.append(imageio.imread(file_path))

            c2w = torch.tensor(frame["transform_matrix"])
            # Convert from OpenGL to OpenCV coordinate system
            c2w[0:3, 1:3] *= -1
            cam_to_worlds.append(c2w)

        self.image_ids = image_ids
        self.cam_to_worlds = cam_to_worlds
        self.images = images

        # all renders have the same intrinsics
        # see also
        # https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/blender_dataparser.py
        image_height, image_width = self.images[0].shape[:2]
        cx = image_width / 2.0
        cy = image_height / 2.0
        fl = 0.5 * image_width / np.tan(0.5 * transforms["camera_angle_x"])
        self.intrinsics = torch.tensor(
            [[fl, 0, cx], [0, fl, cy], [0, 0, 1]], dtype=torch.float32
        )
        self.image_height = image_height
        self.image_width = image_width

        # compute scene scale (as is done in the colmap parser)
        camera_locations = np.stack(self.cam_to_worlds, axis=0)[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = dict(
            K=self.intrinsics,
            camtoworld=self.cam_to_worlds[item],
            image=torch.from_numpy(self.images[item]).float(),
            image_id=item,
        )
        return data
