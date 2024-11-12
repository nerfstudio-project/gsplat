import os
from datetime import datetime
import torch
import sys
from PIL import Image
import random
import imageio
from typing import NamedTuple, Optional
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
import trimesh
import math
import cv2

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return BasicPointCloud(points=positions, colors=colors, normals=None)

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]

        #random.shuffle(frames)
        #frames = frames[:2]

        camtoworld_list = []
        Ks_list = []
        gt_img_list = []
        gt_img_name_list = []

        for idx, frame in enumerate(frames):
            if 'jpg' in frame["file_path"] or 'png' in frame["file_path"]:
                cam_name = os.path.join(path, frame["file_path"])
            else:
                cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            if contents["type"] == "mitsuba":
                c2w[:3, 0:2] *= -1
            else:
                c2w[:3, 1:3] *= -1

            ## get the world-to-camera transform and set R, T
            #w2c = np.linalg.inv(c2w)
            ##R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            #R = w2c[:3,:3]
            #T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            intrinsic_matrix = np.array([
                [fov2focal(FovX, image.size[0]), 0., image.size[0] * 0.5],
                [0., fov2focal(FovY, image.size[1]), image.size[1] * 0.5],
                [0., 0., 1.]
            ], dtype=np.float32)

            camtoworld_list.append(c2w)
            Ks_list.append(intrinsic_matrix)
            gt_img_list.append(image)
            gt_img_name_list.append(image_name)

        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            raise "you should have ply for mitsuba dataset"
        pcd = fetchPly(ply_path)

        return camtoworld_list, Ks_list, gt_img_list, gt_img_name_list, pcd


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def cubemap_to_panorama(path, img_fov90_list, count):
    #img_forward = np.transpose(img_fov90_list[0], (1, 2, 0))[..., ::-1]
    #img_up = np.transpose(img_fov90_list[1], (1, 2, 0))[..., ::-1]
    #img_down = np.transpose(img_fov90_list[2], (1, 2, 0))[..., ::-1]
    #img_left = np.transpose(img_fov90_list[3], (1, 2, 0))[..., ::-1]
    #img_right = np.transpose(img_fov90_list[4], (1, 2, 0))[..., ::-1]
    #img_back = np.transpose(img_fov90_list[5], (1, 2, 0))[..., ::-1]

    img_forward = cv2.imread(os.path.join(path, f'validation/forward/forward_{count}.png'))
    img_up = cv2.imread(os.path.join(path, f'validation/up/up_{count}.png'))
    img_down = cv2.imread(os.path.join(path, f'validation/down/down_{count}.png'))
    img_left = cv2.imread(os.path.join(path, f'validation/left/left_{count}.png'))
    img_right = cv2.imread(os.path.join(path, f'validation/right/right_{count}.png'))
    img_back = cv2.imread(os.path.join(path, f'validation/back/back_{count}.png'))

# Desired output image size (800x800)
    output_width = img_forward.shape[0] * 4
    output_height = img_forward.shape[1] * 4

# Desired field of view
    fov_h_deg = 360  # Horizontal FOV in degrees
    fov_v_deg = 360  # Vertical FOV in degrees

    fov_h_rad = math.radians(fov_h_deg)  # Convert FOV to radians
    fov_v_rad = math.radians(fov_v_deg)

# Create empty output image
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

# Precompute variables
    half_width = output_width / 2.0
    half_height = output_height / 2.0

# Generate grid of pixel coordinates
    x = np.linspace(0, output_width - 1, output_width)
    y = np.linspace(0, output_height - 1, output_height)
    x_grid, y_grid = np.meshgrid(x, y)

# Normalized device coordinates (from -1 to +1)
    nx = (x_grid - half_width) / half_width
    ny = (half_height - y_grid) / half_height  # Invert y-axis for image coordinates

# Compute angles in camera space
    theta = nx * (fov_h_rad / 2)  # Horizontal angle
    phi = ny * (fov_v_rad / 2)    # Vertical angle

# Compute ray directions in camera space
    dir_x = np.sin(theta) * np.cos(phi)
    dir_y = np.sin(phi)
    dir_z = np.cos(theta) * np.cos(phi)

# Normalize the direction vectors
    norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
    dir_x /= norm
    dir_y /= norm
    dir_z /= norm

# Function to sample pixels from an image using bilinear interpolation
    def sample_image(img, u, v):
        img_height, img_width, channels = img.shape

        # Get the integer parts and the fractional parts
        u0 = np.floor(u).astype(np.int32)
        v0 = np.floor(v).astype(np.int32)
        u1 = u0 + 1
        v1 = v0 + 1

        # Clip to valid indices
        u0 = np.clip(u0, 0, img_width - 1)
        u1 = np.clip(u1, 0, img_width - 1)
        v0 = np.clip(v0, 0, img_height - 1)
        v1 = np.clip(v1, 0, img_height - 1)

        # The fractional parts
        fu = u - u0
        fv = v - v0

        # Expand dims to allow broadcasting
        fu = fu[:, None]
        fv = fv[:, None]

        # Get pixel values at the four corners
        Ia = img[v0, u0]  # Shape (N, 3)
        Ib = img[v1, u0]
        Ic = img[v0, u1]
        Id = img[v1, u1]

        # Compute the bilinear interpolation
        wa = (1 - fu) * (1 - fv)
        wb = (1 - fu) * fv
        wc = fu * (1 - fv)
        wd = fu * fv

        # Sum up the weighted contributions
        pixels = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return pixels.astype(np.uint8)

# Determine which face to sample from based on the direction vector components
    abs_dir_x = np.abs(dir_x)
    abs_dir_y = np.abs(dir_y)
    abs_dir_z = np.abs(dir_z)

# Find the maximum component to determine the face
    max_dir = np.maximum.reduce([abs_dir_x, abs_dir_y, abs_dir_z])

# Initialize the masks for each face
    forward_mask = (max_dir == abs_dir_z) & (dir_z > 0)
    back_mask = (max_dir == abs_dir_z) & (dir_z < 0)  # Mask for back face
    right_mask = (max_dir == abs_dir_x) & (dir_x > 0)
    left_mask = (max_dir == abs_dir_x) & (dir_x < 0)
    up_mask = (max_dir == abs_dir_y) & (dir_y > 0)
    down_mask = (max_dir == abs_dir_y) & (dir_y < 0)

# Process each face
    faces = [
        ('forward', forward_mask, img_forward),
        ('back', back_mask, img_back),  # Add the back face processing
        ('right', right_mask, img_right),
        ('left', left_mask, img_left),
        ('up', up_mask, img_up),
        ('down', down_mask, img_down)
    ]

    for face_name, face_mask, img_face in faces:
        if np.any(face_mask):
            # Get the indices of pixels where face_mask is True
            y_indices, x_indices = np.where(face_mask)

            # Extract the direction vectors for these pixels
            dir_x_face = dir_x[face_mask]
            dir_y_face = dir_y[face_mask]
            dir_z_face = dir_z[face_mask]

            # Map to face coordinate system
            if face_name == 'forward':
                dir_img_x = dir_x_face
                dir_img_y = dir_y_face
                dir_img_z = dir_z_face
            elif face_name == 'back':
                dir_img_x = -dir_x_face  # Flip for back face
                dir_img_y = dir_y_face
                dir_img_z = -dir_z_face
            elif face_name == 'right':
                dir_img_x = -dir_z_face
                dir_img_y = dir_y_face
                dir_img_z = dir_x_face
            elif face_name == 'left':
                dir_img_x = dir_z_face
                dir_img_y = dir_y_face
                dir_img_z = -dir_x_face
            elif face_name == 'up':
                dir_img_x = dir_x_face
                dir_img_y = -dir_z_face
                dir_img_z = dir_y_face
            elif face_name == 'down':
                dir_img_x = dir_x_face
                dir_img_y = dir_z_face
                dir_img_z = -dir_y_face

            # Project onto the image plane
            epsilon = 1e-6  # Small value to avoid division by zero
            valid = np.abs(dir_img_z) > epsilon  # Use absolute value to handle near-zero z

            if np.any(valid):
                # Get valid indices
                valid_indices = np.where(valid)[0]

                dir_img_x = dir_img_x[valid]
                dir_img_y = dir_img_y[valid]
                dir_img_z = dir_img_z[valid]

                u_img = dir_img_x / np.abs(dir_img_z)
                v_img = dir_img_y / np.abs(dir_img_z)

                # Convert to pixel coordinates in the input image
                img_height, img_width, _ = img_face.shape

                u = (u_img + 1) * (img_width - 1) / 2.0
                v = (1 - v_img) * (img_height - 1) / 2.0  # Invert y-axis

                # Clip coordinates to image bounds
                u = np.clip(u, 0, img_width - 1)
                v = np.clip(v, 0, img_height - 1)

                # Sample pixels using bilinear interpolation
                pixels = sample_image(img_face, u, v)

                # Assign sampled pixels to output image
                output_image[y_indices[valid_indices], x_indices[valid_indices]] = pixels

    return output_image


def rotation_matrix_to_quaternion(R):
    t = R.trace()
    if t > 0:
        r = torch.sqrt(1 + t)
        s = 0.5 / r
        w = 0.5 * r
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        r = torch.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        s = 0.5 / r
        w = (R[2, 1] - R[1, 2]) * s
        x = 0.5 * r
        y = (R[0, 1] + R[1, 0]) * s
        z = (R[0, 2] + R[2, 0]) * s
    elif R[1, 1] > R[2, 2]:
        r = torch.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        s = 0.5 / r
        w = (R[0, 2] - R[2, 0]) * s
        x = (R[0, 1] + R[1, 0]) * s
        y = 0.5 * r
        z = (R[1, 2] + R[2, 1]) * s
    else:
        r = torch.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        s = 0.5 / r
        w = (R[1, 0] - R[0, 1]) * s
        x = (R[0, 2] + R[2, 0]) * s
        y = (R[1, 2] + R[2, 1]) * s
        z = 0.5 * r
    return torch.tensor([w, x, y, z]).cuda()


def quaternion_to_rotation_matrix(quaternion):
    # Ensure quaternion is normalized
    quaternion = quaternion / torch.norm(quaternion)

    w, x, y, z = quaternion.unbind(-1)

    # Pre-compute repeated values
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z

    # Construct rotation matrix
    R = torch.stack([
        torch.stack([1 - 2 * y2 - 2 * z2, 2 * xy - 2 * wz, 2 * xz + 2 * wy]),
        torch.stack([2 * xy + 2 * wz, 1 - 2 * x2 - 2 * z2, 2 * yz - 2 * wx]),
        torch.stack([2 * xz - 2 * wy, 2 * yz + 2 * wx, 1 - 2 * x2 - 2 * y2])
    ])

    return R
