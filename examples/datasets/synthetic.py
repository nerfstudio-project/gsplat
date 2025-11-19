import os
import json
from pathlib import Path

import imageio.v2 as imageio
from typing import Any, Dict, Optional
from PIL import Image
import numpy as np
import torch

from .normalize import (
    similarity_from_cameras,
    transform_cameras,
)


def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def load_synthetic(data_dir, file, factor: int, id_offset):
    # Built from INRIA's Gaussian Splatting Code
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/8a70a8cd6f0d9c0a14f564844ead2d1147d5a7ac/scene/dataset_readers.py#L179
    camtoworlds = []
    camera_ids = []
    image_names = []
    image_paths = []
    Ks_dict = dict()
    params_dict = dict()
    imsize_dict = dict()  # width, height
    mask_dict = dict()
    with open(os.path.join(data_dir, file)) as json_file:
        contents = json.load(json_file)
        FOVX = contents["camera_angle_x"]

        frames = contents["frames"]

        for idx, frame in enumerate(frames):
            image_path = os.path.join(data_dir, frame["file_path"] + ".png")
            camera_id = idx + id_offset

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            image_name = Path(image_path).stem
            image = Image.open(image_path)

            fx = fov2focal(FOVX, image.width)
            fy = fov2focal(FOVX, image.height)
            cx, cy = 0.5 * image.width, 0.5 * image.height
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            K[:2, :] /= abs(factor)
            Ks_dict[camera_id] = K

            camera_ids.append(camera_id)
            camtoworlds.append(c2w)
            # assume no distortion
            params_dict[camera_id] = np.empty(0, dtype=np.float32)
            imsize_dict[camera_id] = (
                image.width // abs(factor),
                image.height // abs(factor),
            )
            mask_dict[camera_id] = None
            image_names.append(image_name)
            image_paths.append(image_path)
    return (
        camera_ids,
        camtoworlds,
        params_dict,
        imsize_dict,
        mask_dict,
        image_names,
        image_paths,
        Ks_dict,
    )


class Parser:
    """synthetic parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        test_max_res: int = 1600,  # max image side length in pixel for test split
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        # test_every is not needed, as we have a dedicated test set
        self.test_every = test_every

        # Load camera-to-world matrices.
        camtoworlds = []
        camera_ids = []
        image_names = []
        image_paths = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        # load training data
        (
            camera_ids,
            camtoworlds,
            params_dict,
            imsize_dict,
            mask_dict,
            image_names,
            image_paths,
            Ks_dict,
        ) = load_synthetic(data_dir, "transforms_train.json", factor, 0)

        self.train_indices = np.arange(len(image_names))
        train_camera_id_len = len(camera_ids)

        # load test data
        (
            camera_ids_testdata,
            camtoworlds_testdata,
            params_dict_testdata,
            imsize_dict_testdata,
            mask_dict_testdata,
            image_names_testdata,
            image_paths_testdata,
            Ks_dict_testdata,
        ) = load_synthetic(
            data_dir, "transforms_test.json", factor, train_camera_id_len
        )

        # join data
        camera_ids += camera_ids_testdata
        camtoworlds += camtoworlds
        params_dict.update(params_dict_testdata)
        imsize_dict.update(imsize_dict_testdata)
        mask_dict.update(mask_dict_testdata)
        image_names += image_names_testdata
        image_paths += image_paths_testdata
        Ks_dict.update(Ks_dict_testdata)

        self.test_indices = np.arange(len(self.train_indices), len(image_names))

        camtoworlds = np.array(camtoworlds)

        print(
            f"[Parser] {len(image_names)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(image_names) == 0:
            raise ValueError("No images found.")

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # 3D points
        points = None
        points_err = None
        points_rgb = None
        point_indices = dict()

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            transform = T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]

        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        white_background: bool = True,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.white_background = white_background
        if split == "train":
            self.indices = self.parser.train_indices
        else:
            self.indices = self.parser.test_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        camera_id = self.parser.camera_ids[index]
        image = Image.open(self.parser.image_paths[index])
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]
        K = self.parser.Ks_dict[camera_id].copy()

        image = np.array(image)
        bg = (
            np.array([255.0, 255.0, 255.0])
            if self.white_background
            else np.array([0.0, 0.0, 0.0])
        )

        image = image[:, :, :3] * (image[:, :, 3:4] / 255.0) + bg * (
            1 - (image[:, :, 3:4] / 255.0)
        )
        image = image[..., :3]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        return data
