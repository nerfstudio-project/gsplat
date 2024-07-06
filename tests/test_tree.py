import torch

from gsplat.cuda._wrapper import _make_lazy_cuda_func

device = torch.device("cuda:0")


def test_tree_cut():

    leaf_data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], device=device)[
        :, None
    ]
    leaf_mask = torch.tensor(
        [True, True, True, True, True, True, True, True], device=device
    )
    branch_factor = 2
    cut = 0.35

    selected_data, selected_mask = _make_lazy_cuda_func("tree_cut")(
        leaf_data, leaf_mask, branch_factor, cut
    )
    print("selected_data", selected_data)
    print("selected_mask", selected_mask)


if __name__ == "__main__":
    test_tree_cut()
