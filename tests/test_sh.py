import pytest
import torch


device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_sh():
    from gsplat import _torch_impl
    from gsplat import sh

    num_points = 1
    degree = 4
    gt_colors = torch.ones(num_points, 3, device=device) * 0.5
    viewdirs = torch.randn(num_points, 3, device=device)
    viewdirs /= torch.linalg.norm(viewdirs, dim=-1, keepdim=True)
    sh_coeffs = torch.rand(
        num_points, sh.num_sh_bases(degree), 3, device=device, requires_grad=True
    )
    optim = torch.optim.Adam([sh_coeffs], lr=1e-2)

    num_iters = 1000
    for _ in range(num_iters):
        optim.zero_grad()

        # compute PyTorch's color and grad
        check_colors = _torch_impl.compute_sh_color(viewdirs, sh_coeffs)
        check_loss = torch.square(check_colors - gt_colors).mean()
        check_loss.backward()
        check_grad = sh_coeffs.grad.detach()

        optim.zero_grad()

        # compute our colors and grads
        colors = sh.spherical_harmonics(degree, viewdirs, sh_coeffs)
        loss = torch.square(colors - gt_colors).mean()
        loss.backward()
        grad = sh_coeffs.grad.detach()
        optim.step()

        torch.testing.assert_close(check_grad, grad)
        torch.testing.assert_close(check_colors, colors)

    # check final optimized color
    torch.testing.assert_close(check_colors, gt_colors)


if __name__ == "__main__":
    test_sh()
