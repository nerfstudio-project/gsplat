import math
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")
def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(device=device)
    colors = colors[None].repeat(len(viewmats), 1, 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }

def test_projection(test_data):
    from gsplat.cuda._torch_impl import _fully_fused_projection_2dgs

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]
    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True
    _radii, _means2d, _depths, _ray_Ms = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height
    )
    print(f"{_radii.shape=} {_means2d.shape=} {_depths.shape=} {_ray_Ms.shape=}")
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_projection(test_data())
    print("All tests passed.")
