import torch
import imageio
from gsplat import rasterization

device = "cuda:0"

means = torch.randn((100, 3), device=device)
quats = torch.randn((100, 4), device=device)
scales = torch.rand((100, 3), device=device) * 0.1
colors = torch.rand((100, 3), device=device)
opacities = torch.rand((100,), device=device)

viewmats = torch.eye(4, device=device)[None, :, :]
Ks = torch.tensor([[300., 0., 100.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
width, height = 200, 200

renders, alphas, meta = rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, width, height
)
print (colors.shape, alphas.shape)
print (meta.keys())

imageio.imwrite(
    "renders.png", (renders[0] * 255.0).cpu().numpy().astype("uint8")
)