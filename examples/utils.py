import random
import numpy as np
import torch
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps
from fused_ssim import FusedSSIMMap
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

PSNR_EPSILON = 1e-8

class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_dims, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img

def filter_outlier_points(points, colors, nb_neighbors=20, std_ratio=2.0):
    """
    Filter outlier points using statistical outlier removal.
    
    Args:
        points: torch.Tensor of shape [N, 3] - 3D point coordinates
        colors: torch.Tensor of shape [N, 3] - RGB colors corresponding to points
        nb_neighbors: int - number of neighbors to consider for each point
        std_ratio: float - standard deviation ratio threshold
        
    Returns:
        filtered_points: torch.Tensor - points after outlier removal
        filtered_colors: torch.Tensor - colors after outlier removal
        inlier_mask: torch.Tensor - boolean mask indicating inlier points
    """
    if points.shape[0] <= nb_neighbors:
        print(f"Warning: Not enough points ({points.shape[0]}) for outlier filtering (need > {nb_neighbors})")
        return points, colors, torch.ones(points.shape[0], dtype=torch.bool)
    
    # Calculate distances to k nearest neighbors for each point
    k = min(nb_neighbors + 1, points.shape[0])  # +1 because first neighbor is the point itself
    distances = knn(points, k)  # [N, k]
    
    # Calculate mean distance to neighbors (excluding self at index 0)
    mean_distances = distances[:, 1:].mean(dim=1)  # [N]
    
    # Calculate global mean and standard deviation of mean distances
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    
    # Define outlier threshold
    threshold = global_mean + std_ratio * global_std
    
    # Create inlier mask
    inlier_mask = mean_distances <= threshold
    
    # Filter points and colors
    filtered_points = points[inlier_mask]
    filtered_colors = colors[inlier_mask]
    
    num_outliers = (~inlier_mask).sum().item()
    print(f"Filtered out {num_outliers} outlier points ({num_outliers/points.shape[0]*100:.2f}%)")
    print(f"Remaining points: {filtered_points.shape[0]}/{points.shape[0]}")
    
    return filtered_points, filtered_colors, inlier_mask

def get_color(bkgd_color: str):
    if bkgd_color == "green":
        return (0.0, 255.0, 0.0)  # Green background [R, G, B]
    elif bkgd_color == "red":
        return (255.0, 0.0, 0.0)  # Red background [R, G, B]
    elif bkgd_color == "blue":
        return (0.0, 0.0, 255.0)  # Blue background [R, G, B]
    elif bkgd_color == "white":
        return (255.0, 255.0, 255.0)  # White background [R, G, B]
    elif bkgd_color == "black":
        return (0.0, 0.0, 0.0)  # Black background [R, G, B]
    elif bkgd_color == "random":
        return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))  # Random background [R, G, B]
    else:
        return (0.0, 0.0, 0.0)  # Black background [R, G, B]
class Metrics:
    """Comprehensive metrics calculator for image quality evaluation."""
    
    def __init__(self, device: torch.device, lpips_net: str = "alex", data_range: float = 1.0):
        """Initialize all metrics.
        
        Args:
            device: Device to run metrics on
            lpips_net: LPIPS network type ("alex" or "vgg")
            data_range: Data range for PSNR/SSIM (default: 1.0)
        """
        self.device = device
        self.data_range = data_range
        
        # Initialize torchmetrics
        self.psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        
        # Initialize LPIPS metric based on network type
        if lpips_net == "alex":
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(device)
        elif lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(device)
        else:
            raise ValueError(f"Unknown LPIPS network: {lpips_net}")
    
    def _fused_ssim_map(self, img1, img2, padding="same", train=True):
        """Private method for SSIM map calculation."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        assert padding in ["same", "valid"]

        img1 = img1.contiguous()
        map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
        return map
    
    def _get_tight_bbox(self, mask: torch.Tensor, padding: int = 16) -> tuple:
        """Get tight bounding box around masked region with optional padding."""
        # Find the indices where mask is True
        rows = torch.any(mask, dim=2).squeeze(0)  # [H]
        cols = torch.any(mask, dim=1).squeeze(0)  # [W]
        
        # Get bounding box coordinates
        row_indices = torch.where(rows)[0]
        col_indices = torch.where(cols)[0]
        
        if len(row_indices) == 0 or len(col_indices) == 0:
            # No mask found, return full image bounds
            return 0, mask.shape[1], 0, mask.shape[2]
        
        min_row = max(0, row_indices.min().item() - padding)
        max_row = min(mask.shape[1], row_indices.max().item() + 1 + padding)
        min_col = max(0, col_indices.min().item() - padding)
        max_col = min(mask.shape[2], col_indices.max().item() + 1 + padding)
        
        return min_row, max_row, min_col, max_col
    
    def calculate_psnr(
        self,
        colors: torch.Tensor,  # [1, H, W, 3]
        pixels: torch.Tensor,  # [1, H, W, 3]
    ) -> torch.Tensor:
        """Calculate PSNR using torchmetrics."""
        # Convert to [1, 3, H, W] format for torchmetrics
        colors_perm = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
        pixels_perm = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
        
        return self.psnr_metric(colors_perm, pixels_perm)

    def calculate_ssim(
        self,
        colors: torch.Tensor,  # [1, H, W, 3]
        pixels: torch.Tensor,  # [1, H, W, 3]
    ) -> torch.Tensor:
        """Calculate SSIM using torchmetrics."""
        # Convert to [1, 3, H, W] format for torchmetrics
        colors_perm = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
        pixels_perm = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
        
        return self.ssim_metric(colors_perm, pixels_perm)

    def calculate_lpips(
        self,
        colors: torch.Tensor,  # [1, H, W, 3]
        pixels: torch.Tensor,  # [1, H, W, 3]
    ) -> torch.Tensor:
        """Calculate LPIPS using torchmetrics."""
        # Convert to [1, 3, H, W] format for torchmetrics
        colors_perm = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
        pixels_perm = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
        
        return self.lpips_metric(colors_perm, pixels_perm)

    def calculate_masked_psnr1(
        self,
        colors_p: torch.Tensor, # [1, H, W, 3]
        pixels_p: torch.Tensor, # [1, H, W, 3]
        masks: torch.Tensor,    # [1, H, W], bool
    ) -> torch.Tensor: # [1]
        """Calculate PSNR by averaging per-pixel PSNR values in masked region."""
        target_mask = masks.unsqueeze(-1)  # Shape: [1, H, W, 1]
        # Alternative approach: flatten and use boolean indexing
        # Expand mask to match RGB channels: [1, H, W, 1] -> [1, H, W, 3]
        target_mask_expanded = target_mask.expand_as(colors_p)  # [1, H, W, 3]
        
        # Flatten and extract only masked pixels
        colors_flat = colors_p.reshape(-1)  # [1*H*W*3]
        pixels_flat = pixels_p.reshape(-1)  # [1*H*W*3]
        mask_flat = target_mask_expanded.reshape(-1)  # [1*H*W*3]
        
        colors_p_masked = colors_flat[mask_flat]  # [N_masked_pixels]
        pixels_p_masked = pixels_flat[mask_flat]  # [N_masked_pixels]
        per_sample_mses = F.mse_loss(colors_p_masked, pixels_p_masked)
        
        # Correct PSNR formula: PSNR = 10 * log10(MAX^2 / MSE) where MAX=1.0
        per_sample_psnrs_gpu = -10.0 * torch.log10(per_sample_mses + PSNR_EPSILON) # Shape: [1]
        return per_sample_psnrs_gpu

    def calculate_masked_psnr2(
        self,
        colors_p: torch.Tensor, # [1,H,W,3]
        pixels_p: torch.Tensor, # [1,H,W,3]
        masks: torch.Tensor,    # [1,H,W]
    ) -> torch.Tensor: # [1]
        """Calculate PSNR over masked region by computing overall MSE first, then converting to PSNR."""
        pred_rgb_for_psnr = colors_p        # Shape: [1,H,W,3]
        target_rgb_for_psnr = pixels_p      # Shape: [1,H,W,3]
        target_mask = masks.unsqueeze(-1)  # Shape: [1,H,W,1]
        
        # Only compute errors for masked pixels
        pred_rgb_masked = pred_rgb_for_psnr * target_mask # [1,H,W,3]
        target_rgb_masked = target_rgb_for_psnr * target_mask # [1,H,W,3]
        
        # Compute squared errors only on valid pixels
        sq_errors_unreduced = F.mse_loss(pred_rgb_masked, target_rgb_masked, reduction="none") # [1,H,W,3]
        sq_errors_masked = sq_errors_unreduced * target_mask # Zero out invalid pixels # [1,H,W,3]

        # Compute per-sample MSE by averaging over valid pixels only
        # mask_counts = target_mask.sum(dim=[1,2,3])  # [1] - number of valid RGB values
        mask_counts = (target_mask * 3).sum(dim=[1,2,3])  # [1] - number of valid RGB channel values
        per_sample_mse_sums = sq_errors_masked.sum(dim=[1,2,3])  # [1] - sum of errors per sample
        
        # Check if we have any masked pixels
        if mask_counts.item() > 0:
            per_sample_mses = per_sample_mse_sums / mask_counts
            
            # Correct PSNR formula: PSNR = 10 * log10(MAX^2 / MSE) where MAX=1.0
            per_sample_psnrs_gpu = -10.0 * torch.log10(per_sample_mses + PSNR_EPSILON) # Shape: [1]
        else:
            # No masked pixels
            per_sample_psnrs_gpu = torch.tensor([-1.0], device=colors_p.device)
        
        return per_sample_psnrs_gpu

    def calculate_masked_ssim(
        self,
        colors: torch.Tensor,  # [1, H, W, 3]
        pixels: torch.Tensor,  # [1, H, W, 3]  
        masks: torch.Tensor,   # [1, H, W], bool
    ) -> torch.Tensor:
        """Calculate masked SSIM using per-pixel SSIM map and averaging only over masked region."""
        # Convert to [1, 3, H, W] format for fused_ssim_map
        colors_perm = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
        pixels_perm = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
        
        # Calculate per-pixel SSIM map
        per_pixel_ssim = self._fused_ssim_map(colors_perm, pixels_perm, padding="same")  # [1, 3, H, W] or [1, H, W]
        
        # Handle different output shapes from fused_ssim_map
        if per_pixel_ssim.dim() == 4:  # [1, 3, H, W]
            # Average across channels to get [1, H, W]
            per_pixel_ssim = per_pixel_ssim.mean(dim=1)  # [1, H, W]
        
        # Calculate masked SSIM - average only over masked pixels
        if masks.sum() > 0:
            masked_ssim = per_pixel_ssim[masks].mean()
        else:
            masked_ssim = torch.tensor(0.0, device=colors.device)
        
        return masked_ssim

    def calculate_masked_se_rmse(
        self,
        colors: torch.Tensor,  # [1, H, W, 3]
        pixels: torch.Tensor,  # [1, H, W, 3]  
        masks: torch.Tensor,   # [1, H, W], bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate masked sum of squared errors (SE) and root mean squared errors (RMSE)."""
        if masks.sum() == 0:
            # No valid pixels in mask
            return torch.tensor(0.0, device=colors.device), torch.tensor(0.0, device=colors.device)
        
        # Expand mask to match image dimensions [1, H, W, 3]
        mask_expanded = masks.unsqueeze(-1).expand_as(colors)  # [1, H, W, 3]
        
        # Calculate squared differences
        squared_diff = (colors - pixels) ** 2  # [1, H, W, 3]
        
        # Apply mask and sum over all dimensions (spatial and channel)
        masked_squared_diff = squared_diff * mask_expanded.float()  # [1, H, W, 3]
        
        # Sum of squared errors (SE) - unnormalized sum over all masked pixels and channels
        se = masked_squared_diff.sum()  # Scalar tensor
        
        # Root mean squared error (RMSE) - normalized by number of valid elements
        num_valid_elements = mask_expanded.sum()  # Total number of valid pixel-channel combinations
        mse = se / num_valid_elements  # Mean squared error
        rmse = torch.sqrt(mse)  # Root mean squared error
        
        return se, rmse

    def calculate_cropped_lpips(
        self,
        colors: torch.Tensor,  # [1, H, W, 3]
        pixels: torch.Tensor,  # [1, H, W, 3]
        masks: torch.Tensor,   # [1, H, W], bool
        method: str = "cropped"  # "cropped", "zeroed", or "conditional"
    ) -> tuple:
        """Calculate LPIPS over cropped masked regions using different strategies."""
        # Calculate mask coverage ratio
        total_pixels = masks.numel()
        mask_pixels = masks.sum().item()
        coverage_ratio = mask_pixels / total_pixels
        
        # Determine if LPIPS is reliable (mask covers at least 10% of image)
        is_reliable = coverage_ratio >= 0.1
        
        if method == "cropped" and mask_pixels > 0:
            # Extract tight bounding box around masked region
            min_row, max_row, min_col, max_col = self._get_tight_bbox(masks, padding=1)
            
            # Crop both images to bounding box
            colors_cropped = colors[:, min_row:max_row, min_col:max_col, :]  # [1, H', W', 3]
            pixels_cropped = pixels[:, min_row:max_row, min_col:max_col, :]  # [1, H', W', 3]
            
            # Ensure minimum size for LPIPS (LPIPS needs reasonable spatial resolution)
            h_crop, w_crop = colors_cropped.shape[1], colors_cropped.shape[2]
            if h_crop < 32 or w_crop < 32:
                # Fall back to full image if crop is too small
                method = "zeroed"
            else:
                # Convert to [1, 3, H', W'] format for LPIPS
                colors_perm = colors_cropped.permute(0, 3, 1, 2)
                pixels_perm = pixels_cropped.permute(0, 3, 1, 2)
                
                # Calculate LPIPS on cropped region
                lpips_val = self.lpips_metric(colors_perm, pixels_perm)
                return lpips_val, coverage_ratio, True  # Cropped LPIPS is more reliable
        
        if method == "conditional" and not is_reliable:
            # Return NaN for unreliable cases
            return torch.tensor(float('nan'), device=colors.device), coverage_ratio, False
        
        # Fall back to zeroed method (original implementation)
        mask_expanded = masks.unsqueeze(-1).expand_as(colors)  # [1, H, W, 3]
        
        colors_masked = colors * mask_expanded.float()
        pixels_masked = pixels * mask_expanded.float()
        
        # Convert to [1, 3, H, W] format for LPIPS
        colors_perm = colors_masked.permute(0, 3, 1, 2)  # [1, 3, H, W]
        pixels_perm = pixels_masked.permute(0, 3, 1, 2)  # [1, 3, H, W]
        
        # Calculate LPIPS (this will include zero regions)
        lpips_val = self.lpips_metric(colors_perm, pixels_perm)
        
        return lpips_val, coverage_ratio, is_reliable
    
    def calculate_all_metrics(
        self, 
        culled_image: torch.Tensor,  # [1, H, W, 3]
        gt_image: torch.Tensor,      # [1, H, W, 3]
        gt_mask: torch.Tensor,       # [1, H, W], bool
        measure_lpips: bool = True
    ) -> dict:
        """Calculate all quality metrics for a pair of images.
        
        Args:
            culled_image: Predicted/culled image [1, H, W, 3]
            gt_image: Ground truth image [1, H, W, 3]
            gt_mask: Binary mask [1, H, W], bool
            measure_lpips: Whether to calculate LPIPS (can be slow)
            
        Returns:
            Dictionary containing all metric values
        """
        results = {}
        
        # Basic metrics (PSNR, SSIM, LPIPS)
        results['psnr'] = self.calculate_psnr(culled_image, gt_image).item()
        results['ssim'] = self.calculate_ssim(culled_image, gt_image).item()
        
        if measure_lpips:
            results['lpips'] = self.calculate_lpips(culled_image, gt_image).item()
        else:
            results['lpips'] = 0.0
        
        # Masked metrics
        results['masked_psnr'] = self.calculate_masked_psnr1(culled_image, gt_image, gt_mask).item()
        results['masked_ssim'] = self.calculate_masked_ssim(culled_image, gt_image, gt_mask).item()
        
        # Masked SE and RMSE
        masked_se_val, masked_rmse_val = self.calculate_masked_se_rmse(culled_image, gt_image, gt_mask)
        results['masked_se'] = masked_se_val.item()
        results['masked_rmse'] = masked_rmse_val.item()
        
        # Cropped LPIPS
        if measure_lpips:
            cropped_lpips_result = self.calculate_cropped_lpips(
                culled_image, gt_image, gt_mask, method="cropped"
            )
            cropped_lpips_val, mask_coverage, lpips_reliable = cropped_lpips_result
            results['cropped_lpips'] = cropped_lpips_val.item()
            results['mask_coverage'] = mask_coverage
            results['lpips_reliable'] = lpips_reliable
        else:
            results['cropped_lpips'] = 0.0
            results['mask_coverage'] = 0.0
            results['lpips_reliable'] = False
        
        # Additional mask statistics
        results['num_valid_pixels'] = gt_mask.sum().item()
        
        return results