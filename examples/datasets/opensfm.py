import os
import numpy as np
import collections
from typing_extensions import assert_never

import math
import json
from pyproj import Proj

import cv2
import imageio
from typing import Dict, List, Any, Optional
import torch
from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params", "panorama"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids", "diff_ref"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def angle_axis_to_quaternion(angle_axis: np.ndarray):
    angle = np.linalg.norm(angle_axis)

    x = angle_axis[0] / angle
    y = angle_axis[1] / angle
    z = angle_axis[2] / angle

    qw = math.cos(angle / 2.0)
    qx = x * math.sqrt(1 - qw * qw)
    qy = y * math.sqrt(1 - qw * qw)
    qz = z * math.sqrt(1 - qw * qw)

    return np.array([qw, qx, qy, qz])

def angle_axis_and_angle_to_quaternion(angle, axis):
    half_angle = angle / 2.0
    sin_half_angle = math.sin(half_angle)
    return np.array([
        math.cos(half_angle),
        axis[0] * sin_half_angle,
        axis[1] * sin_half_angle,
        axis[2] * sin_half_angle
    ])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([w, x, y, z])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

class Parser:
    """Parser for Opensfm data formatted similarly to the COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        # Extract data from reconstructions.

        reconstructions = self.load_reconstructions(data_dir)

        self._parse_reconstructions(reconstructions)

    def _parse_reconstructions(self, reconstructions: List[Dict]):
        """Parse reconstructions data to extract camera information, extrinsics, and 3D points."""
        self.cameras, self.images = read_opensfm(reconstructions)
        self.points3D, self.colors, self.errors = read_opensfm_points3D(reconstructions)
        points = self.points3D.astype(np.float32)
        self.colors = self.colors.astype(np.uint8)
        self.errors = self.errors.astype(np.float32)

        # Extract extrinsic matrices in world-to-camera format.
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()
        mask_dict = dict()
        point_indices = {img.name: img.point3D_ids for img in self.images.values()}
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        for img in self.images.values():
            # Extract rotation and translation vectors.
            rot = img.qvec2rotmat()
            trans = img.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = img.camera_id
            camera_ids.append(camera_id)

            # Camera intrinsics
            cam = self.cameras[img.camera_id]
            type_ = cam.model
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
                fx, fy, cx, cy = cam.params[0], cam.params[0], cam.params[1], cam.params[2]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                K[:2, :] /= self.factor
                Ks_dict[img.camera_id] = K

                # Distortion parameters
                params_dict[img.camera_id] = np.append(cam.params[3:5], np.array([0, 0]))
                imsize_dict[img.camera_id] = (cam.width // self.factor, cam.height // self.factor)
                mask_dict[camera_id] = None
                camera_ids.append(img.camera_id)
            elif type_ == 5 or type_ == "SPHERICAL":
                params = np.empty(0, dtype=np.float32)
                camtype = "spherical"
                K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                Ks_dict[img.camera_id] = K
                params_dict[img.camera_id] = params
                imsize_dict[img.camera_id] = (cam.width, cam.height)
                mask_dict[camera_id] = None
                camera_ids.append(img.camera_id)

        w2c_mats = np.stack(w2c_mats, axis=0)
        
        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Normalize the world space if needed.
        if self.normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        # Set instance variables.

        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_rgb = self.colors  # np.ndarray, (num_points, 3)
        self.points_err = self.errors  # np.ndarray, (num_points, 1)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = fx * x1 * r + width // 2
                mapy = fy * y1 * r + height // 2

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    def load_reconstructions(self, data_dir):
        reconstructions_file = os.path.join(data_dir, 'reconstruction.json')
        with open(reconstructions_file, 'r') as f:
            reconstructions = json.load(f)
        return reconstructions

class Dataset:
    """A simple dataset class for OpensfmLoaderParser."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.images))  # Use images from parser
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        img = self.parser.images[index]
        camera_id = img.camera_id
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        camtoworld = self.parser.camtoworlds[index]

        image_path = os.path.join(self.parser.data_dir, "images/" + img.name)  # Update with actual image path
        image = imageio.imread(image_path)[..., :3]

        # Undistort if necessary
        params = self.parser.params_dict[camera_id]
        if len(params) > 0:
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }

        if self.load_depths:
            # Load depth data (dummy implementation, replace with actual depth loading logic if available)
            depths = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)  # Placeholder for actual depth loading
            data["depths"] = torch.from_numpy(depths).float()

        return data

def read_opensfm(reconstructions):
    """Extracts camera and image information from OpenSfM reconstructions."""
    images = {}
    i = 0
    reference_lat_0 = reconstructions[0]["reference_lla"]["latitude"]
    reference_lon_0 = reconstructions[0]["reference_lla"]["longitude"]
    reference_alt_0 = reconstructions[0]["reference_lla"]["altitude"]
    e2u_zone = int(divmod(reference_lon_0, 6)[0]) + 31
    e2u_conv = Proj(proj='utm', zone=e2u_zone, ellps='WGS84')
    reference_x_0, reference_y_0 = e2u_conv(reference_lon_0, reference_lat_0)
    if reference_lat_0 < 0:
        reference_y_0 += 10000000
    
    cameras = {}
    camera_names = {}
    cam_id = 1
    
    for reconstruction in reconstructions:
        # Parse cameras.
        for i, camera in enumerate(reconstruction["cameras"]):
            camera_name = camera
            camera_info = reconstruction["cameras"][camera]
            if camera_info['projection_type'] in ['spherical', 'equirectangular']:
                camera_id = 0
                model = "SPHERICAL"
                width = reconstruction["cameras"][camera]["width"]
                height = reconstruction["cameras"][camera]["height"]
                f = width / 4 / 2
                params = np.array([f, width, height])
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params, panorama=True)
                camera_names[camera_name] = camera_id
            elif reconstruction["cameras"][camera]['projection_type'] == "perspective":
                model = "SIMPLE_PINHOLE"
                width = reconstruction["cameras"][camera]["width"]
                height = reconstruction["cameras"][camera]["height"]
                f = reconstruction["cameras"][camera]["focal"] * width
                k1 = reconstruction["cameras"][camera]["k1"]
                k2 = reconstruction["cameras"][camera]["k2"]
                params = np.array([f, width / 2, height / 2, k1, k2])
                camera_id = cam_id
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params, panorama=False)
                camera_names[camera_name] = camera_id
                cam_id += 1
        
        reference_lat = reconstruction["reference_lla"]["latitude"]
        reference_lon = reconstruction["reference_lla"]["longitude"]
        reference_alt = reconstruction["reference_lla"]["altitude"]
        reference_x, reference_y = e2u_conv(reference_lon, reference_lat)
        if reference_lat < 0:
            reference_y += 10000000
        
        for shot in reconstruction["shots"]:
            translation = reconstruction["shots"][shot]["translation"]
            rotation = reconstruction["shots"][shot]["rotation"]
            qvec = angle_axis_to_quaternion(rotation)
            diff_ref_x = reference_x - reference_x_0
            diff_ref_y = reference_y - reference_y_0
            diff_ref_alt = reference_alt - reference_alt_0
            tvec = np.array([translation[0], translation[1], translation[2]])
            diff_ref = np.array([diff_ref_x, diff_ref_y, diff_ref_alt])
            camera_name = reconstruction["shots"][shot]["camera"]
            camera_id = camera_names.get(camera_name, 0)
            image_id = i
            image_name = shot
            xys = np.array([0, 0])
            point3D_ids = np.array([0, 0])
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids, diff_ref=diff_ref)
            i += 1
    return cameras, images

def read_opensfm_points3D(reconstructions):
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    for reconstruction in reconstructions:
        num_points = num_points + len(reconstruction["points"])

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    reference_lat_0 = reconstructions[0]["reference_lla"]["latitude"]
    reference_lon_0 = reconstructions[0]["reference_lla"]["longitude"]
    reference_alt_0 = reconstructions[0]["reference_lla"]["altitude"]
    e2u_zone=int(divmod(reference_lon_0, 6)[0])+31
    e2u_conv=Proj(proj='utm', zone=e2u_zone, ellps='WGS84')
    reference_x_0, reference_y_0 = e2u_conv(reference_lon_0, reference_lat_0)
    if reference_lat_0<0:
        reference_y=reference_y+10000000
    for reconstruction in reconstructions:
        reference_lat = reconstruction["reference_lla"]["latitude"]
        reference_lon = reconstruction["reference_lla"]["longitude"]
        reference_alt = reconstruction["reference_lla"]["altitude"]
        reference_x, reference_y = e2u_conv(reference_lon, reference_lat)
        for i in (reconstruction["points"]):
            color = (reconstruction["points"][i]["color"])
            coordinates = (reconstruction["points"][i]["coordinates"])
            xyz = np.array([coordinates[0] + reference_x - reference_x_0, coordinates[1] + reference_y - reference_y_0, coordinates[2] - reference_alt + reference_alt_0])
            rgb = np.array([color[0], color[1], color[2]])
            error = np.array(0)
            xyzs[count] = xyz
            rgbs[count] = rgb
            errors[count] = error
            count += 1
    return xyzs, rgbs, errors