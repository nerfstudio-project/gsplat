import torch
import tqdm
from torch import Tensor

from gsplat.cuda._wrapper import _make_lazy_cuda_func

device = torch.device("cuda:0")


def _merge(data: Tensor, mask: Tensor):
    # data: (B, M, D), mask: (B, M)
    # return: (B, D)
    # Simply sum all the valid data
    return (data * mask[..., None]).sum(dim=1)


def _is_larger(data: Tensor, mask: Tensor, threshold: float):
    # data: (..., D) mask: (...,), threshold: float
    # return: (...,) boolen mask
    # Simply use the first element of the data to compare with the threshold
    return (data[..., 0] > threshold) & mask


def tree_cut_impl(
    leaf_data: Tensor,
    leaf_mask: Tensor,
    branch_factor: int,
    threshold: float,
    verbose: bool = False,
):
    """Find the cut in a tree, where:

    The node above the cut is larger than the threshold, and the node below the cut is
    smaller than the threshold.

    Or

    The node below the cut is a leaf node and it is above the threshold.

    .. note::
        We expect the tree input to this function is a balanced tree with a fixed
        branching factor.

    Args:
        leaf_data: The data of the leaf nodes. Shape: (N, D)
        leaf_mask: The mask of the leaf nodes that indicates which nodes are valid. Shape: (N,)
        branch_factor: Branching factor of the tree. E.g. 2, 4, 8 ...
        threshold: The threshold to cut the tree.

    Returns:
        data: The selected node data that satisfies the condition. Shape: (M, D)
    """
    assert leaf_data.dim() == 2, leaf_data.dim()
    assert leaf_mask.dim() == 1, leaf_mask.dim()
    assert len(leaf_data) == len(leaf_mask), (len(leaf_data), len(leaf_mask))

    n_leaf = len(leaf_data)  # n_leaf must be a power of branch_factor

    selected_data = []

    # Bottom-up build the tree (up-sweep)
    n_nodes = n_leaf
    node_data = leaf_data
    node_mask = leaf_mask
    is_leaf = True
    while n_nodes > 1:
        # check if the nodes are larger than the threshold.
        is_node_larger = _is_larger(node_data, node_mask, threshold)

        # reshape data into [n_parents, branch_factor, D]
        n_parents = n_nodes // branch_factor
        group_data = node_data.reshape(n_parents, branch_factor, -1)
        group_mask = node_mask.reshape(n_parents, branch_factor)
        is_node_larger = is_node_larger.reshape(n_parents, branch_factor)

        # merge children into parent
        data = _merge(group_data, group_mask)  # [n_parents, D]
        mask = group_mask.any(dim=1)  # [n_parents]

        # check if the parent is larger than the threshold. if so, we find a cut
        # and the children which are smaller than the threshold are selected.
        is_parent_larger = _is_larger(data, mask, threshold)  # [n_parents]

        # sel is [n_parents, branch_factor]
        if is_leaf:
            sel = is_parent_larger[:, None] & group_mask
        else:
            sel = is_parent_larger[:, None] & ~is_node_larger & group_mask
        sel = sel.reshape(n_parents * branch_factor)  # flatten it

        selected_data.append(node_data[sel])
        if verbose & sel.any():
            print("Selected nodes that satisfied the cut:", node_data[sel])

        node_data = data
        node_mask = mask
        n_nodes = n_parents
        is_leaf = False

        if is_parent_larger.all():
            # All the parent nodes are larger than the threshold. We
            # can stop the up-sweep.
            break

    selected_data = torch.cat(selected_data, dim=0)
    return selected_data


def test_tree_cut():
    torch.manual_seed(42)

    # number of leaf nodes should be a power of the branch factor
    # i.e., 4096 = banch_factor^n
    leaf_data = torch.rand(4096, 1, device=device)
    leaf_mask = torch.rand(4096, device=device) > 0.9
    branch_factor = 2  # or 8 ...
    cut = 0.5

    # CUDA impl. (only forward pass)
    selected_data, selected_mask = _make_lazy_cuda_func("tree_cut")(
        leaf_data, leaf_mask, branch_factor, cut
    )
    selected_data = selected_data[selected_mask]

    # PyTorch impl.
    _selected_data = tree_cut_impl(
        leaf_data=leaf_data,
        leaf_mask=leaf_mask,
        branch_factor=branch_factor,
        threshold=cut,
        verbose=False,
    )

    # testing (the order of the two impl is not guaranteed to be the same, so we sort them first)
    sorted_selected_data = torch.sort(selected_data, dim=0, descending=False).values
    _sorted_selected_data = torch.sort(_selected_data, dim=0, descending=False).values
    torch.testing.assert_close(_sorted_selected_data, sorted_selected_data)


if __name__ == "__main__":
    test_tree_cut()
