import torch
from diff_rast._torch_impl import compute_sh_color
from diff_rast.sh import num_sh_bases, SphericalHarmonics


if __name__ == "__main__":
    device = torch.device("cuda:0")
    num_points = 1
    degree = 4

    gt_colors = torch.ones(num_points, 3, device=device) * 0.5
    viewdirs = torch.randn(num_points, 3, device=device)
    viewdirs /= torch.linalg.norm(viewdirs, dim=-1, keepdim=True)
    # print("viewdirs", viewdirs)
    sh_coeffs = torch.rand(
        num_points, num_sh_bases(degree), 3, device=device, requires_grad=True
    )
    # print("sh_coeffs", sh_coeffs)
    optim = torch.optim.Adam([sh_coeffs], lr=1e-2)

    num_iters = 100
    for _ in range(num_iters):
        optim.zero_grad()
        # compute python version's color and grad
        check_colors = compute_sh_color(viewdirs, sh_coeffs)
        check_loss = torch.square(check_colors - gt_colors).mean()
        check_loss.backward()
        check_grad = sh_coeffs.grad.detach()

        optim.zero_grad()
        # compute our colors and grads
        colors = SphericalHarmonics.apply(degree, viewdirs, sh_coeffs)
        loss = torch.square(colors - gt_colors).mean()
        loss.backward()
        grad = sh_coeffs.grad.detach()
        optim.step()

        diff_colors = (check_colors - colors).detach()
        diff_grad = (check_grad - grad).detach()

        print(f"LOSS {loss.item():.2e}")
        # print("colors", colors)
        # print("check_colors", check_colors)
        # print("diff_colors", diff_colors)
        # print("grad", grad)
        # print("check grad", check_grad)
        # print("diff grad", diff_grad)
        print(f"colors min {colors.min().item():.2e} max {colors.max().item():.2e}")
        print(
            f"check_colors min {check_colors.min().item():.2e} max {check_colors.max().item():.2e}"
        )
        print(
            f"diff_colors min {diff_colors.min().item():.2e} max {diff_colors.max().item():.2e}"
        )
        print(f"grad min {grad.min().item():.2e} max {grad.max().item():.2e}")
        print(
            f"check_grad min {check_grad.min().item():.2e} max {check_grad.max().item():.2e}"
        )
        print(
            f"diff_grad min {diff_grad.min().item():.2e} max {diff_grad.max().item():.2e}"
        )
