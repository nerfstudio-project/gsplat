import pytest
import torch


device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_compare_binding_to_pytorch():
    from gsplat._torch_impl import compute_cov2d_bounds as _compute_cov2d_bounds
    from gsplat import compute_cov2d_bounds

    torch.manual_seed(42)

    num_cov2ds = 100

    _covs2d = torch.rand(
        (num_cov2ds, 2, 2), dtype=torch.float32, device=device, requires_grad=True
    )
    covs2d = torch.stack(
        [
            torch.triu(_covs2d)[:, 0, 0],
            torch.triu(_covs2d)[:, 0, 1],
            torch.triu(_covs2d)[:, 1, 1],
        ],
        dim=-1,
    )

    conic, radii = compute_cov2d_bounds(covs2d)
    _conic, _radii, _mask = _compute_cov2d_bounds(_covs2d)

    radii = radii.squeeze(-1)

    atol = 5e-4
    rtol = 1e-5
    torch.testing.assert_close(conic[_mask], _conic[_mask], atol=atol, rtol=rtol)
    torch.testing.assert_close(radii[_mask], _radii[_mask], atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_compare_binding_to_pytorch()
