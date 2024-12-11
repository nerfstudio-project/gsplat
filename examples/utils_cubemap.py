import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import tqdm
import torch.nn.functional as F
import torchvision
import math

def homogenize(X: torch.Tensor):
    assert X.ndim == 2
    assert X.shape[1] in (2, 3)
    return torch.cat(
        (X, torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)), dim=1
    )

def dehomogenize(X: torch.Tensor):
    assert X.ndim == 2
    assert X.shape[1] in (3, 4)
    return X[:, :-1] / X[:, -1:]

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def generate_pts(viewpoint_cam, boundary_scale=4, sample_resolution=20):
    width = viewpoint_cam['image'].shape[1]
    height = viewpoint_cam['image'].shape[0]
    sample_width = int(width / sample_resolution)
    sample_height = int(height / sample_resolution)
    K = viewpoint_cam['K'].cuda()
    width = viewpoint_cam['fisheye_image'].shape[1]
    height = viewpoint_cam['fisheye_image'].shape[0]
    width = int(width * boundary_scale)
    height = int(height * boundary_scale)
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    i, j = np.meshgrid(
        np.linspace(0, width, sample_width),
        np.linspace(0, height, sample_height),
        indexing="ij",
    )
    i = i.T
    j = j.T

    P_sensor = (
        torch.from_numpy(np.stack((i, j), axis=-1))
        .to(torch.float32)
        .cuda()
    )
    P_sensor_hom = homogenize(P_sensor.reshape((-1, 2)))
    P_view_insidelens_direction_hom = (K.inverse() @ P_sensor_hom.T).T
    P_view_insidelens_direction = dehomogenize(P_view_insidelens_direction_hom)

    return P_sensor, P_view_insidelens_direction


def init_from_coeff(coeff, ref_points):
    r = torch.sqrt(torch.sum(ref_points**2, dim=-1, keepdim=True))
    inv_r = 1 / (r + 1e-5)
    theta = torch.atan(r)
    if len(coeff) == 4:
        ref_points = ref_points * (inv_r * (theta + coeff[0] * theta**3 + coeff[1] * theta**5 + coeff[2] * theta**7 + coeff[3] * theta**9))
    else:
        assert "bug"
    print(f"using coeff: {coeff}")

    return ref_points

def plot_points(ref_points, path):
    p1 = ref_points.clone().reshape(-1, 2)
    plt.figure(figsize=(int(ref_points.shape[1]/4), int(ref_points.shape[0]/4)))
    x = p1[:, 0].detach().cpu().numpy()  # Convert tensor to numpy for plotting
    y = p1[:, 1].detach().cpu().numpy()
    plt.scatter(x, y)
    plt.title('2D Points Plot')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.xlim(p1[:, 0].min().item() - 0.1, p1[:, 0].max().item() + 0.1)
    plt.ylim(p1[:, 1].min().item() - 0.1, p1[:, 1].max().item() + 0.1)
    plt.grid(True)
    plt.savefig(path)

def init_from_colmap(viewpoint_cam, coeff, result_dir, lens_net, optimizer_lens_net, scheduler_lens_net, iresnet_lr=1e-7):
    P_sensor, P_view_insidelens_direction = generate_pts(viewpoint_cam, boundary_scale=1.5, sample_resolution=40)
    P_view_outsidelens_direction = P_view_insidelens_direction
    camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
    ref_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
    plot_points(ref_points, os.path.join(result_dir, f"ref1_gt.png"))
    ref_points = init_from_coeff(coeff, ref_points)
    inf_mask = torch.isinf(ref_points)
    nan_mask = torch.isnan(ref_points)
    ref_points[inf_mask] = 0
    ref_points[nan_mask] = 0
    plot_points(ref_points, os.path.join(result_dir, f"ref1.png"))

    P_sensor, P_view_insidelens_direction = generate_pts(viewpoint_cam, boundary_scale=4., sample_resolution=40)
    P_view_outsidelens_direction = P_view_insidelens_direction
    camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
    ref_points1 = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
    plot_points(ref_points1, os.path.join(result_dir, f"ref2_gt.png"))
    ref_points1 = init_from_coeff(coeff, ref_points1)
    inf_mask = torch.isinf(ref_points1)
    nan_mask = torch.isnan(ref_points1)
    ref_points1[inf_mask] = 0
    ref_points1[nan_mask] = 0
    plot_points(ref_points1, os.path.join(result_dir, f"ref2.png"))
    combine = torch.cat((ref_points, ref_points1), dim=0)
    plot_points(combine, os.path.join(result_dir, f"ref1_2.png"))

    P_sensor0, P_view_insidelens_direction0 = generate_pts(viewpoint_cam, boundary_scale=1.5, sample_resolution=40)
    P_sensor1, P_view_insidelens_direction1 = generate_pts(viewpoint_cam, boundary_scale=4., sample_resolution=40)
    P_sensor =torch.cat((P_sensor0, P_sensor1), dim=0)
    P_view_insidelens_direction =torch.cat((P_view_insidelens_direction0, P_view_insidelens_direction1), dim=0)

    progress_bar_ires = tqdm.tqdm(range(0, 5000), desc="Init Iresnet")
    for i in range(5000):
        #P_view_insidelens_direction -> network -> compare with combine
        #combine -> network -> compare with P_view_insidelens_direction
        P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=True)
        #P_view_outsidelens_direction = lens_net.forward(combine.reshape(-1, 2), sensor_to_frustum=True)
        control_points = P_view_outsidelens_direction
        inf_mask = torch.isinf(P_view_outsidelens_direction)
        nan_mask = torch.isnan(P_view_outsidelens_direction)
        P_view_outsidelens_direction[inf_mask] = 0
        P_view_outsidelens_direction[nan_mask] = 0
        loss = ((P_view_outsidelens_direction - combine.reshape(-1, 2))**2).mean()
        #loss = ((P_view_outsidelens_direction - P_view_insidelens_direction)**2).mean()
        progress_bar_ires.set_postfix({"loss": f"{loss.item():.7f}"})
        progress_bar_ires.update(1)
        loss.backward()
        optimizer_lens_net.step()
        optimizer_lens_net.zero_grad(set_to_none = True)
        scheduler_lens_net.step()

        if i % 2000 == 1:
            control_points_np = P_view_outsidelens_direction.cpu().detach().numpy()
            combine_np = combine.reshape(-1, 2).cpu().detach().numpy()
            #combine_np = P_view_insidelens_direction.reshape(-1, 2).cpu().detach().numpy()
            plt.figure(figsize=(10, 6))
            plt.scatter(control_points_np[:, 0], control_points_np[:, 1], color='blue')
            plt.scatter(combine_np[:, 0], combine_np[:, 1], color='red')
            plt.savefig(os.path.join(result_dir, f"loss_{i}.png"))
            plt.close()

    progress_bar_ires.close()

    #for param_group in optimizer_lens_net.param_groups:
    #    param_group['lr'] = iresnet_lr
    #print(f"The learning rate is reset to {param_group['lr']}")

def generate_control_pts(viewpoint_cam, control_point_sample_scale=32, flow_scale=[2., 2.]):
    width = viewpoint_cam['image'].shape[2]
    height = viewpoint_cam['image'].shape[1]
    sample_width = int(width / control_point_sample_scale)
    sample_height = int(height/ control_point_sample_scale)
    K = viewpoint_cam['K'].cuda().squeeze(0)
    width = viewpoint_cam['fisheye_image'].shape[2]
    height = viewpoint_cam['fisheye_image'].shape[1]
    width = int(width * flow_scale[0])
    height = int(height * flow_scale[1])
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    i, j = np.meshgrid(
        np.linspace(0, width, sample_width),
        np.linspace(0, height, sample_height),
        indexing="ij",
    )
    i = i.T
    j = j.T
    P_sensor = (
        torch.from_numpy(np.stack((i, j), axis=-1))
        .to(torch.float32)
        .cuda()
    )
    P_sensor_hom = homogenize(P_sensor.reshape((-1, 2)))
    P_view_insidelens_direction_hom = (torch.inverse(K) @ P_sensor_hom.T).T
    P_view_insidelens_direction = dehomogenize(P_view_insidelens_direction_hom)

    return P_sensor, P_view_insidelens_direction

def getProjectionMatrix(znear, zfar, fovX, fovY):
    if torch.is_tensor(fovX) and torch.is_tensor(fovY):
        tanHalfFovY = torch.tan((fovY / 2))
        tanHalfFovX = torch.tan((fovX / 2))
    else:
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def center_crop(tensor, target_height, target_width):
    _, _, height, width = tensor.size()

    # Calculate the starting coordinates for the crop
    start_y = (height - target_height) // 2
    start_x = (width - target_width) // 2

    # Create a grid for the interpolation
    grid_y, grid_x = torch.meshgrid(torch.linspace(start_y, start_y + target_height - 1, target_height),
                                    torch.linspace(start_x, start_x + target_width - 1, target_width))
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).to(tensor.device)

    # Normalize grid to [-1, 1]
    grid = 2.0 * grid / torch.tensor([width - 1, height - 1]).cuda() - 1.0
    grid = grid.permute(0, 1, 2, 3).expand(tensor.size(0), target_height, target_width, 2)

    # Perform the interpolation
    cropped_tensor = F.grid_sample(tensor, grid, align_corners=True)

    return cropped_tensor

def apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint_cam, image, projection_matrix, flow_scale=[2., 2.], apply2gt=False, debug_grid=None, update_debug_grid=False):
    #P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=True)
    #torch.save(P_view_outsidelens_direction, '/home/yd428/gsplat/examples/results/netflix/paul_office_crop/P_view_outsidelens_direction.pt')
    #torch.save(P_view_outsidelens_direction, '/home/yd428/gsplat/examples/results/netflix/paul_garden_scale2/P_view_outsidelens_direction.pt')
    #torch.save(P_view_outsidelens_direction, '/home/yd428/gsplat/examples/results/netflix/paul_office_crop_deblur_optpose/P_view_outsidelens_direction.pt')
    if debug_grid != None:
        P_view_outsidelens_direction = debug_grid
    else:
        P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=apply2gt)
        #import pdb;pdb.set_trace()
        #print(9)

    camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
    control_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]

    flow = control_points @ projection_matrix[:2, :2]

    if apply2gt:
        flow = F.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(int(viewpoint_cam['image'].shape[1]*2.), int(viewpoint_cam['image'].shape[2]*2.)), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
        gt_image = F.grid_sample(viewpoint_cam['fisheye_image'].cuda().permute(0, 3, 1, 2)/255, flow.unsqueeze(0), mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(0)
        mask = (~((gt_image[0]<0.00001) & (gt_image[1]<0.00001)).unsqueeze(0)).float()
        return gt_image, mask, flow

    flow = F.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(int(viewpoint_cam['fisheye_image'].shape[1]*flow_scale[0]), int(viewpoint_cam['fisheye_image'].shape[2]*flow_scale[1])), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
    image = F.grid_sample(
        image.permute(0, 3, 1, 2),
        flow.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    image = center_crop(image, viewpoint_cam['fisheye_image'].shape[1], viewpoint_cam['fisheye_image'].shape[2])
    mask = (~((image[0][0]==0.0000) & (image[0][1]==0.0000)).unsqueeze(0)).float()
    if not update_debug_grid:
        return image, mask, flow
    else:
        return image, mask, flow, P_view_outsidelens_direction.detach()

def rotate_camera(camtoworlds, deg_x, deg_y, deg_z):
    R = camtoworlds[:3, :3].cpu().numpy()  # World-to-camera rotation matrix
    T = camtoworlds[:3, -1:].cpu().numpy()  # World-to-camera translation matrix
    camera_center_world = -np.dot(R.T, T)
    R_camera_to_world = R.T  # Inverse of the rotation matrix in world-to-camera space

    theta_x = np.deg2rad(deg_x)  # Convert degrees to radians
    theta_y = np.deg2rad(deg_y)  # Convert degrees to radians
    theta_z = np.deg2rad(deg_z)  # Convert degrees to radians
    right_camera = R_camera_to_world[:, 0]   # Right (x-axis)
    up_camera = R_camera_to_world[:, 1]      # Up (y-axis)
    forward_camera = R_camera_to_world[:, 2] # Forward (z-axis)
    Ry = Rotation.from_rotvec(theta_y * up_camera).as_matrix()
    Rx = Rotation.from_rotvec(theta_x * right_camera).as_matrix()
    Rz = Rotation.from_rotvec(theta_z * forward_camera).as_matrix()

    R_camera_to_world_new = np.dot(Rz, np.dot(Rx, np.dot(Ry, R_camera_to_world)))
    R_new = R_camera_to_world_new.T
    T_new = -np.dot(R_new, camera_center_world)

    new_camtoworlds = np.eye(4, dtype=np.float32)
    new_camtoworlds[:3, :3] = R_new
    new_camtoworlds[:3, -1:] = T_new

    return torch.from_numpy(new_camtoworlds).cuda()

def generate_pts_up_down_left_right(width, height, K, shift_width=0, shift_height=0, sample_rate=1):
    i, j = np.meshgrid(np.linspace(0 + shift_width * width, width + shift_width * width, width//sample_rate), np.linspace(0 + shift_height * height, height + shift_height * height, height//sample_rate), indexing="ij")
    i = i.T
    j = j.T
    P_sensor = (
        torch.from_numpy(np.stack((i, j), axis=-1))
        .to(torch.float32)
        .cuda()
    )
    P_sensor_hom = homogenize(P_sensor.reshape((-1, 2)))
    P_view_insidelens_direction_hom = (torch.inverse(K) @ P_sensor_hom.T).T
    P_view_insidelens_direction = dehomogenize(P_view_insidelens_direction_hom)
    return P_view_insidelens_direction


def interpolate_with_control(control_r, control_theta, r):
    """
    Interpolates control_theta values at specified r locations using linear interpolation.

    Parameters:
    - control_r (torch.Tensor): 1D tensor of reference points for interpolation.
    - control_theta (torch.Tensor): Function values at control_r points.
    - r (torch.Tensor): Target points at which to interpolate control_theta.

    Returns:
    - torch.Tensor: Interpolated values at each point in r, with the same shape as r.
    """
    # Flatten control_r for easy indexing
    control_r_flat = control_r.squeeze()

    # Find indices of the two nearest control_r points for each value in r
    indices = torch.searchsorted(control_r_flat, r.squeeze(), right=True)
    indices = torch.clamp(indices, 1, len(control_r_flat) - 1)

    # Get the lower and upper neighbors for each element in r
    low_indices = indices - 1
    high_indices = indices

    # Fetch the corresponding values from control_r and control_theta
    r_low = control_r_flat[low_indices]
    r_high = control_r_flat[high_indices]
    theta_low = control_theta[low_indices]
    theta_high = control_theta[high_indices]

    # Calculate weights and perform linear interpolation
    weights = (r.squeeze() - r_low) / (r_high - r_low)
    interpolated_theta = (1 - weights).unsqueeze(1) * theta_low + weights.unsqueeze(1) * theta_high

    # Ensure output matches the shape of r
    return interpolated_theta.view_as(r)

def apply_flow_up_down_left_right(width, height, K, rays, rays_residual, img, types="forward", is_fisheye=False, control_r=None, control_theta=None, cubemap_net_threshold=1.57):
    r = torch.sqrt(torch.sum(rays**2, dim=-1, keepdim=True))
    if is_fisheye:
        #r[r > cubemap_net_threshold] = cubemap_net_threshold
        #theta = torch.tan(r) + interpolate_with_control(control_r, control_theta, r)
        theta = torch.tan(r)
        inv_r = 1 / (r + 1e-5)

        scale = theta * inv_r
        rays_dis = scale * rays
    else:
        rays_dis = rays

    rays_dis_hom = homogenize(rays_dis)

    if types == 'left':
        x = rays_dis_hom[:, 0]  # First column (x)
        y = rays_dis_hom[:, 1]  # Second column (y)
        z = rays_dis_hom[:, 2]  # Third column (z)
        P_left = torch.stack((-z / x, -y / x), dim=1)  # Shape: [N, 2]
        rays_dis_hom = homogenize(P_left)
    elif types == 'right':
        x = rays_dis_hom[:, 0]  # First column (x)
        y = rays_dis_hom[:, 1]  # Second column (y)
        z = rays_dis_hom[:, 2]  # Third column (z)
        P_right = torch.stack((-z / x, y / x), dim=1)  # Shape: [N, 2]
        rays_dis_hom = homogenize(P_right)
    elif types == 'up':
        x = rays_dis_hom[:, 0]  # First column (x)
        y = rays_dis_hom[:, 1]  # Second column (y)
        z = rays_dis_hom[:, 2]  # Third column (z)
        P_up = torch.stack((-x / y, -z / y), dim=1)  # Shape: [N, 2]
        rays_dis_hom = homogenize(P_up)

    elif types == 'down':
        x = rays_dis_hom[:, 0]  # First column (x)
        y = rays_dis_hom[:, 1]  # Second column (y)
        z = rays_dis_hom[:, 2]  # Third column (z)
        P_down = torch.stack((x / y, -z / y), dim=1)  # Shape: [N, 2]
        rays_dis_hom = homogenize(P_down)

    rays_dis_inside = dehomogenize((K @ rays_dis_hom.T).T).reshape(height, width, 2)

    # apply flow field
    x_coords = rays_dis_inside[..., 0]  # Shape: [800, 800]
    y_coords = rays_dis_inside[..., 1]  # Shape: [800, 800]
    x_coords_norm = (x_coords / (img.shape[2] - 1)) * 2 - 1
    y_coords_norm = (y_coords / (img.shape[1] - 1)) * 2 - 1
    grid = torch.stack((x_coords_norm, y_coords_norm), dim=-1)  # Shape: [800, 800, 2]
    grid = grid.unsqueeze(0)

    img_forward_batch = img.unsqueeze(0)  # Shape: [1, 3, 800, 800]
    distorted_img = F.grid_sample(
        img_forward_batch,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)
    distorted_img = distorted_img.squeeze(0)  # Shape: [3, 800, 800]

    if types == 'forward':
        return distorted_img, img, control_theta
    return distorted_img, img

def mask_half(image: torch.Tensor, direction: str = "left") -> torch.Tensor:
    _, h, w = image.shape

    # Create a mask with ones for the unmasked area and zeros for the masked area
    mask = torch.ones_like(image)

    if direction == "right":
        # Mask the left half
        mask[:, :, :w // 2] = 0
    elif direction == "left":
        # Mask the right half
        mask[:, :, w // 2:] = 0
    elif direction == "down":
        # Mask the upper half
        mask[:, :h // 2, :] = 0
    elif direction == "up":
        # Mask the lower half
        mask[:, h // 2:, :] = 0
    else:
        raise ValueError("Invalid direction. Choose from 'left', 'right', 'up', 'down'.")

    # Apply the mask to the image (differentiable)
    masked_image = image * mask

    return masked_image, mask

def generate_circular_mask(image_shape: torch.Size, radius: int) -> torch.Tensor:
    """
    Generates a circular mask based on the provided radius.

    Args:
    - image_shape (torch.Size): The shape of the image (C, H, W).
    - radius (int): The radius of the circular mask.

    Returns:
    - torch.Tensor: A circular mask with shape (C, H, W) where the area inside the
      radius from the center is 1 and outside is 0.
    """
    _, h, w = image_shape

    # Create a coordinate grid centered at the image center
    y_center, x_center = h // 2, w // 2
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Calculate the distance of each point from the center
    dist_from_center = torch.sqrt((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2)

    # Create the circular mask
    mask = (dist_from_center <= radius).float()

    # Expand the mask to match the shape of the image (C, H, W)
    mask = mask.unsqueeze(0).repeat(3, 1, 1)

    return mask

def render_cubemap(width, height, new_width, new_height, fov90_width, fov90_height, Ks, camtoworlds, cfg, sh_degree_to_use, image_ids, masks, cubemap_net, rasterize_splats, cubemap_net_threshold, render_pano=False):
    mask_fov90 = torch.zeros((1, height, width), dtype=torch.float32).cuda()
    mask_fov90[:, height//2 - int(fov90_height//2) - 2:height//2 + int(fov90_height//2) + 2, width//2 - int(fov90_width//2) - 2:width//2 + int(fov90_width//2) + 2] = 1

    camtoworlds_up = rotate_camera(camtoworlds[0].inverse(), 90, 0, 0).unsqueeze(0).inverse()
    camtoworlds_down = rotate_camera(camtoworlds[0].inverse(), -90, 0, 0).unsqueeze(0).inverse()
    camtoworlds_right = rotate_camera(camtoworlds[0].inverse(), 0, 90, 0).unsqueeze(0).inverse()
    camtoworlds_left = rotate_camera(camtoworlds[0].inverse(), 0, -90, 0).unsqueeze(0).inverse()
    if render_pano:
        camtoworlds_back = rotate_camera(camtoworlds[0].inverse(), 0, 180, 0).unsqueeze(0).inverse()
        img_perspective_list = []
    info_list = []
    renders_forward, _, info_forward = rasterize_splats(camtoworlds=camtoworlds, Ks=Ks, width=new_width, height=new_height, sh_degree=sh_degree_to_use, near_plane=cfg.near_plane, far_plane=cfg.far_plane, image_ids=image_ids, render_mode="RGB+ED" if cfg.depth_loss else "RGB", masks=masks)
    renders_up, _, info_up = rasterize_splats(camtoworlds=camtoworlds_up, Ks=Ks, width=new_width, height=new_height, sh_degree=sh_degree_to_use, near_plane=cfg.near_plane, far_plane=cfg.far_plane, image_ids=image_ids, render_mode="RGB+ED" if cfg.depth_loss else "RGB", masks=masks)
    renders_down, _, info_down = rasterize_splats(camtoworlds=camtoworlds_down, Ks=Ks, width=new_width, height=new_height, sh_degree=sh_degree_to_use, near_plane=cfg.near_plane, far_plane=cfg.far_plane, image_ids=image_ids, render_mode="RGB+ED" if cfg.depth_loss else "RGB", masks=masks)
    renders_left, _, info_left = rasterize_splats(camtoworlds=camtoworlds_left, Ks=Ks, width=new_width, height=new_height, sh_degree=sh_degree_to_use, near_plane=cfg.near_plane, far_plane=cfg.far_plane, image_ids=image_ids, render_mode="RGB+ED" if cfg.depth_loss else "RGB", masks=masks)
    renders_right, _, info_right = rasterize_splats(camtoworlds=camtoworlds_right, Ks=Ks, width=new_width, height=new_height, sh_degree=sh_degree_to_use, near_plane=cfg.near_plane, far_plane=cfg.far_plane, image_ids=image_ids, render_mode="RGB+ED" if cfg.depth_loss else "RGB", masks=masks)
    if render_pano:
        renders_back, _, _ = rasterize_splats(camtoworlds=camtoworlds_back, Ks=Ks, width=new_width, height=new_height, sh_degree=sh_degree_to_use, near_plane=cfg.near_plane, far_plane=cfg.far_plane, image_ids=image_ids, render_mode="RGB+ED" if cfg.depth_loss else "RGB", masks=masks)
        img_perspective_list.append(renders_forward)
        img_perspective_list.append(renders_up)
        img_perspective_list.append(renders_down)
        img_perspective_list.append(renders_left)
        img_perspective_list.append(renders_right)
        img_perspective_list.append(renders_back)
    info_list.append(info_forward)
    info_list.append(info_up)
    info_list.append(info_down)
    info_list.append(info_left)
    info_list.append(info_right)

    #torchvision.utils.save_image(renders_forward.permute(0, 3, 1, 2)[0] * mask_fov90, os.path.join(cfg.result_dir, f'forward.png'))
    #torchvision.utils.save_image(renders_up.permute(0, 3, 1, 2)[0] * mask_fov90, os.path.join(cfg.result_dir, f'up.png'))
    #torchvision.utils.save_image(renders_down.permute(0, 3, 1, 2)[0] * mask_fov90, os.path.join(cfg.result_dir, f'down.png'))
    #torchvision.utils.save_image(renders_left.permute(0, 3, 1, 2)[0] * mask_fov90, os.path.join(cfg.result_dir, f'left.png'))
    #torchvision.utils.save_image(renders_right.permute(0, 3, 1, 2)[0] * mask_fov90, os.path.join(cfg.result_dir, f'right.png'))


    rays_4_faces = generate_pts_up_down_left_right(new_width, new_height, Ks[0])
    rays_residual_4_faces = generate_pts_up_down_left_right(new_width, new_height, Ks[0])
    img_list = []
    #control_r = torch.linspace(0, cubemap_net_threshold, 10000).unsqueeze(1).cuda()
    #control_theta = cubemap_net.forward(control_r, sensor_to_frustum=True)
    control_theta, control_r = None, None

    img_fish_forward, img_perspective, control_theta = apply_flow_up_down_left_right(width, height, Ks[0], rays_4_faces, rays_residual_4_faces, renders_forward.permute(0, 3, 1, 2)[0]*mask_fov90, types="forward", is_fisheye=True, control_r=control_r, control_theta=control_theta, cubemap_net_threshold=cubemap_net_threshold)
    img_list.append(img_fish_forward)
    img_fish_up, img_perspective = apply_flow_up_down_left_right(width, height, Ks[0], rays_4_faces, rays_residual_4_faces, renders_up.permute(0, 3, 1, 2)[0]*mask_fov90, types="up", is_fisheye=True, control_r=control_r, control_theta=control_theta, cubemap_net_threshold=cubemap_net_threshold)
    img_fish_up, half_mask = mask_half(img_fish_up, 'up')
    img_list.append(img_fish_up)
    img_fish_down, img_perspective = apply_flow_up_down_left_right(width, height, Ks[0], rays_4_faces, rays_residual_4_faces, renders_down.permute(0, 3, 1, 2)[0]*mask_fov90, types="down", is_fisheye=True, control_r=control_r, control_theta=control_theta, cubemap_net_threshold=cubemap_net_threshold)
    img_fish_down, half_mask = mask_half(img_fish_down, 'down')
    img_list.append(img_fish_down)
    img_fish_left, img_perspective = apply_flow_up_down_left_right(width, height, Ks[0], rays_4_faces, rays_residual_4_faces, renders_left.permute(0, 3, 1, 2)[0]*mask_fov90, types="left", is_fisheye=True, control_r=control_r, control_theta=control_theta, cubemap_net_threshold=cubemap_net_threshold)
    img_fish_left, half_mask = mask_half(img_fish_left, 'left')
    img_list.append(img_fish_left)
    img_fish_right, img_perspective = apply_flow_up_down_left_right(width, height, Ks[0], rays_4_faces, rays_residual_4_faces, renders_right.permute(0, 3, 1, 2)[0]*mask_fov90, types="right", is_fisheye=True, control_r=control_r, control_theta=control_theta, cubemap_net_threshold=cubemap_net_threshold)
    img_fish_right, half_mask = mask_half(img_fish_right, 'right')
    img_list.append(img_fish_right)
    if render_pano:
        img_fish_back, img_perspective = apply_flow_up_down_left_right(width, height, Ks[0], rays_4_faces, rays_residual_4_faces, renders_back.permute(0, 3, 1, 2)[0]*mask_fov90, types="right", is_fisheye=True, control_r=control_r, control_theta=control_theta, cubemap_net_threshold=cubemap_net_threshold)

    #torchvision.utils.save_image(img_fish_forward, os.path.join(cfg.result_dir, f'forward_fish.png'))
    #torchvision.utils.save_image(img_fish_up, os.path.join(cfg.result_dir, f'up_fish.png'))
    #torchvision.utils.save_image(img_fish_down, os.path.join(cfg.result_dir, f'down_fish.png'))
    #torchvision.utils.save_image(img_fish_left, os.path.join(cfg.result_dir, f'left_fish.png'))
    #torchvision.utils.save_image(img_fish_right, os.path.join(cfg.result_dir, f'right_fish.png'))

    if render_pano:
        return img_list, info_list, img_perspective_list
    return img_list, info_list

