import pytest
import traceback
import torch
from torch.func import vjp  # type: ignore

from gsplat import _torch_impl
from gsplat.project_gaussians import project_gaussians
import gsplat.cuda as _C

torch.manual_seed(42)

device = torch.device("cuda:0")


def projection_matrix(fx, fy, W, H, n=0.01, f=1000.0):
    return torch.tensor(
        [
            [2.0 * fx / W, 0.0, 0.0, 0.0],
            [0.0, 2.0 * fy / H, 0.0, 0.0],
            [0.0, 0.0, (f + n) / (f - n), -2 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )


def check_close(a, b, atol=1e-5, rtol=1e-5):
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    except AssertionError:
        traceback.print_exc()
        diff = torch.abs(a - b).detach()
        print(f"{diff.max()=} {diff.mean()=}")
        import ipdb

        ipdb.set_trace()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_project_gaussians_forward():
    num_points = 100

    means3d = torch.randn((num_points, 3), device=device, requires_grad=True)
    scales = torch.rand((num_points, 3), device=device) + 0.2
    glob_scale = 0.1
    quats = torch.randn((num_points, 4), device=device)
    quats /= torch.linalg.norm(quats, dim=-1, keepdim=True)

    H, W = 512, 512
    cx, cy = W / 2, H / 2
    # 90 degree FOV
    fx, fy = W / 2, W / 2
    clip_thresh = 0.01
    viewmat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )
    viewmat[:3, :3] = _torch_impl.quat_to_rotmat(torch.randn(4))
    BLOCK_SIZE = 16

    (
        xys,
        depths,
        radii,
        conics,
        compensation,
        num_tiles_hit,
        cov3d,
    ) = project_gaussians(
        means3d,
        scales,
        glob_scale,
        quats,
        viewmat,
        fx,
        fy,
        cx,
        cy,
        H,
        W,
        BLOCK_SIZE,
        clip_thresh,
    )
    masks = num_tiles_hit > 0

    with torch.no_grad():
        (
            _cov3d,
            _,
            _xys,
            _depths,
            _radii,
            _conics,
            _compensation,
            _num_tiles_hit,
            _masks,
        ) = _torch_impl.project_gaussians_forward(
            means3d,
            scales,
            glob_scale,
            quats,
            viewmat,
            (fx, fy, cx, cy),
            (W, H),
            BLOCK_SIZE,
            clip_thresh,
        )

    check_close(masks, _masks, atol=1e-5, rtol=1e-5)
    check_close(cov3d, _cov3d)
    check_close(xys, _xys)
    check_close(depths, _depths)
    check_close(radii, _radii)
    check_close(conics, _conics)
    check_close(compensation, _compensation)
    check_close(num_tiles_hit, _num_tiles_hit)
    print("passed project_gaussians_forward test")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_project_gaussians_backward():
    num_points = 100

    means3d = torch.randn((num_points, 3), device=device, requires_grad=True)
    scales = torch.rand((num_points, 3), device=device) + 0.2
    glob_scale = 0.1
    quats = torch.randn((num_points, 4), device=device)
    quats /= torch.linalg.norm(quats, dim=-1, keepdim=True)

    H, W = 512, 512
    cx, cy = W / 2, H / 2
    # 90 degree FOV
    fx, fy = W / 2, W / 2
    clip_thresh = 0.01
    viewmat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )
    viewmat[:3, :3] = _torch_impl.quat_to_rotmat(torch.randn(4))

    BLOCK_SIZE = 16

    (
        cov3d,
        cov2d,
        xys,
        depths,
        radii,
        conics,
        compensation,
        _,
        masks,
    ) = _torch_impl.project_gaussians_forward(
        means3d,
        scales,
        glob_scale,
        quats,
        viewmat,
        (fx, fy, cx, cy),
        (W, H),
        BLOCK_SIZE,
        clip_thresh,
    )

    # Test backward pass

    v_xys = torch.randn_like(xys)
    v_depths = torch.randn_like(depths)
    v_conics = torch.randn_like(conics)
    # compensation gradients
    v_compensation = torch.randn_like(compensation)
    v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat = _C.project_gaussians_backward(
        num_points,
        means3d,
        scales,
        glob_scale,
        quats,
        viewmat,
        fx,
        fy,
        cx,
        cy,
        H,
        W,
        cov3d,
        radii,
        conics,
        compensation,
        v_xys,
        v_depths,
        v_conics,
        v_compensation,
    )

    def scale_rot_to_cov3d_partial(scale, quat):
        """
        scale (*, 3), quat (*, 3) -> cov3d (upper tri) (*, 6)
        """
        cov3d = _torch_impl.scale_rot_to_cov3d(scale, glob_scale, quat)
        i, j = torch.triu_indices(3, 3)
        cov3d_triu = cov3d[..., i, j]
        return cov3d_triu

    def project_cov3d_ewa_partial(mean3d, cov3d):
        """
        mean3d (*, 3), cov3d (upper tri) (*, 6) -> cov2d (upper tri) (*, 3)
        """
        tan_fovx = 0.5 * W / fx
        tan_fovy = 0.5 * H / fy

        cov3d_mat = torch.zeros(*cov3d.shape[:-1], 3, 3, device=device)
        i, j = torch.triu_indices(3, 3)
        cov3d_mat[..., i, j] = cov3d
        cov3d_mat[..., [1, 2, 2], [0, 0, 1]] = cov3d[..., [1, 2, 4]]
        cov2d, _ = _torch_impl.project_cov3d_ewa(
            mean3d, cov3d_mat, viewmat, fx, fy, tan_fovx, tan_fovy
        )
        ii, jj = torch.triu_indices(2, 2)
        return cov2d[..., ii, jj]

    def compute_cov2d_bounds_partial(cov2d):
        """
        cov2d (upper tri) (*, 3) -> conic (upper tri) (*, 3)
        """
        cov2d_mat = torch.zeros(*cov2d.shape[:-1], 2, 2, device=device)
        i, j = torch.triu_indices(2, 2)
        cov2d_mat[..., i, j] = cov2d
        cov2d_mat[..., 1, 0] = cov2d[..., 1]
        conic, _, _ = _torch_impl.compute_cov2d_bounds(cov2d_mat)
        return conic

    def compute_compensation_partial(cov2d):
        """
        cov2d (upper tri) (*, 3) -> compensation (*, 1)
        """
        cov2d_mat = torch.zeros(*cov2d.shape[:-1], 2, 2, device=device)
        i, j = torch.triu_indices(2, 2)
        cov2d_mat[..., i, j] = cov2d
        cov2d_mat[..., 1, 0] = cov2d[..., 1]
        return _torch_impl.compute_compensation(cov2d_mat)

    def project_pix_partial(mean3d):
        """
        mean3d (*, 3) -> xy (*, 2)
        """
        p_view, _ = _torch_impl.clip_near_plane(mean3d, viewmat, clip_thresh)
        return _torch_impl.project_pix((fx, fy), p_view, (cx, cy))

    def compute_depth_partial(mean3d):
        """
        mean3d (*, 3) -> depth (*)
        """
        p_view, _ = _torch_impl.clip_near_plane(mean3d, viewmat, clip_thresh)
        depth = p_view[..., 2]
        return depth

    _, vjp_scale_rot_to_cov3d = vjp(scale_rot_to_cov3d_partial, scales, quats)  # type: ignore
    _, vjp_project_cov3d_ewa = vjp(project_cov3d_ewa_partial, means3d, cov3d)  # type: ignore
    _, vjp_compute_cov2d_bounds = vjp(compute_cov2d_bounds_partial, cov2d)  # type: ignore
    _, vjp_compute_compensation = vjp(compute_compensation_partial, cov2d)  # type: ignore
    _, vjp_project_pix = vjp(project_pix_partial, means3d)  # type: ignore
    _, vjp_compute_depth = vjp(compute_depth_partial, means3d)  # type: ignore

    _v_cov2d = vjp_compute_cov2d_bounds(v_conics)[0]
    _v_cov2d = _v_cov2d + vjp_compute_compensation(v_compensation)[0]
    _v_mean3d_cov2d, _v_cov3d = vjp_project_cov3d_ewa(_v_cov2d)
    _v_mean3d_xy = vjp_project_pix(v_xys)[0]
    _v_mean3d_depth = vjp_compute_depth(v_depths)[0]
    _v_mean3d = _v_mean3d_cov2d + _v_mean3d_xy + _v_mean3d_depth
    _v_scale, _v_quat = vjp_scale_rot_to_cov3d(_v_cov3d)

    atol = 5e-4
    rtol = 1e-5
    check_close(v_cov2d, _v_cov2d, atol=atol, rtol=rtol)
    check_close(v_cov3d, _v_cov3d, atol=atol, rtol=rtol)
    check_close(v_mean3d[:, :], _v_mean3d[:, :], atol=atol, rtol=rtol)
    check_close(v_scale, _v_scale, atol=atol, rtol=rtol)
    check_close(v_quat, _v_quat, atol=atol, rtol=rtol)
    print("passed project_gaussians_backward test")


if __name__ == "__main__":
    test_project_gaussians_forward()
    test_project_gaussians_backward()
