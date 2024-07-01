import os
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import Tensor


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _distributed_worker(
    world_rank: int,
    world_size: int,
    fn: Callable,
    args: Any,
    local_rank: Optional[int] = None,
    verbose: bool = False,
) -> bool:
    if local_rank is None:  # single Node
        local_rank = world_rank
    if verbose:
        print("Distributed worker: %d / %d" % (world_rank + 1, world_size))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=world_rank
        )
        # Dump collection that participates all ranks.
        # This initializes the communicator required by `batch_isend_irecv`.
        # See: https://github.com/pytorch/pytorch/pull/74701
        _ = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(_, 0)
    fn(local_rank, world_rank, world_size, args)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if verbose:
        print("Job Done for worker: %d / %d" % (world_rank + 1, world_size))
    return True


def cli(fn: Callable, args: Any, verbose: bool = False) -> bool:
    assert torch.cuda.is_available(), "CUDA device is required!"
    if "OMPI_COMM_WORLD_SIZE" in os.environ:  # multi-node
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])  # dist.get_world_size()
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])  # dist.get_rank()
        return _distributed_worker(
            world_rank, world_size, fn, args, local_rank, verbose
        )

    world_size = torch.cuda.device_count()
    distributed = world_size > 1

    if distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        process_context = torch.multiprocessing.spawn(
            _distributed_worker,
            args=(world_size, fn, args, None, verbose),
            nprocs=world_size,
            join=False,
        )
        try:
            process_context.join()
        except KeyboardInterrupt:
            # this is important.
            # if we do not explicitly terminate all launched subprocesses,
            # they would continue living even after this main process ends,
            # eventually making the OD machine unusable!
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    if verbose:
                        print("terminating process " + str(i) + "...")
                    process.terminate()
                process.join()
                if verbose:
                    print("process " + str(i) + " finished")
        return True
    else:
        return _distributed_worker(0, 1, fn=fn, args=args)


def broadcast(data: Tensor, rank: int, lengths: List[int]) -> List[Tensor]:
    """broadcast data across GPUs.
    Note this function also works for a single GPU, in which case the `rank=0` and
    `lengths=[data.shape[0]]`.
    Args:
        data (Tensor): The data that lives in this GPU. Support shape [N, ...]
        rank (int): This GPU rank.
        lengths (List[int]): A list indicates the size of the data (i.e. N) in each GPU.
    Returns:
        List[Tensor]: A list of tensors, in the order of rank0, rank1, ... rankN.
        The size of each Tensor cooresponds to the `lengths` argument.
    """
    assert data.shape[0] == lengths[rank], (
        f"the size of the data ({data.shape[0]}) "
        f"does not match the size specified in lengths ({lengths[rank]})"
    )

    Ds = data.shape[1:]  # dimension of the data.
    dtype = data.dtype
    device = data.device

    def _empty(shape):
        return torch.empty(shape, dtype=dtype, device=device)

    # Ops to recv data from all other GPUs (except itself).
    data_recvs = [
        _empty((l, *Ds)) if r != rank else None for r, l in enumerate(lengths)
    ]
    op_recvs = [
        torch.distributed.P2POp(torch.distributed.irecv, data_recvs[r], r, tag=r)
        for r, l in enumerate(lengths)
        if r != rank and l > 0
    ]

    # Ops to send data to all other GPUs (except itself).
    if lengths[rank] > 0:
        op_sends = [
            torch.distributed.P2POp(torch.distributed.isend, data, r, tag=rank)
            for r, _ in enumerate(lengths)
            if r != rank
        ]
    else:
        op_sends = []

    # Execute the ops.
    ops = op_recvs + op_sends
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # Set the data for itself.
    data_recvs[rank] = data

    # The returned data is a list of tensors, in the order of rank0, rank1, ...
    return data_recvs


def sendrecv(
    data: Tensor, rank: int, lengths: List[int], rank_dst: int
) -> Optional[List[Tensor]]:
    """Send/Recv data across GPUs to a specific rank.
    Note this function also works for a single GPU, in which case the `rank=0` and
    `lengths=[data.shape[0]]`.
    Args:
        data (Tensor): The data that lives in this GPU. Support shape [N, ...]
        rank (int): This GPU rank.
        lengths (List[int]): A list indicates the size of the data (i.e. N) in each GPU.
        rank_dst (Optional[int]): The rank that the result will be sent to.
    Returns:
        List[Tensor]: A list of tensors, in the order of rank0, rank1, ... rankN.
        The size of each Tensor cooresponds to the `lengths` argument. Only the rank
        specified by `rank_dst` will have the results. Others will return None.
    """
    assert data.shape[0] == lengths[rank], (
        f"the size of the data ({data.shape[0]}) "
        f"does not match the size specified in lengths ({lengths[rank]})"
    )

    Ds = data.shape[1:]  # dimension of the data.
    dtype = data.dtype
    device = data.device

    def _empty(shape):
        return torch.empty(shape, dtype=dtype, device=device)

    # collect the data from all ranks only to the rank specified by `rank_dst`.
    if rank == rank_dst:
        # `rank_dst` only have recv ops.
        data_recvs = [
            _empty((l, *Ds)) if r != rank else None for r, l in enumerate(lengths)
        ]
        op_recvs = [
            torch.distributed.P2POp(torch.distributed.irecv, data_recvs[r], r, tag=r)
            for r, l in enumerate(lengths)
            if r != rank and l > 0
        ]
        op_sends = []
    else:
        # All other ranks only have send ops.
        data_recvs = None
        op_recvs = []
        if lengths[rank] > 0:
            op_sends = [
                torch.distributed.P2POp(
                    torch.distributed.isend, data, rank_dst, tag=rank
                )
            ]
        else:
            op_sends = []

    # Execute the ops.
    ops = op_recvs + op_sends
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # Set the data for itself.
    if rank == rank_dst:
        data_recvs[rank] = data

    # The returned data is a list of tensors, in the order of rank0, rank1, ..., or None.
    return data_recvs


def alpha_compositing(
    colors: Tensor,  # [..., n_gpus, 3]
    accs: Tensor,  # [..., n_gpus, 1]
    depths: Tensor,  # [..., n_gpus, 1]
    terminates: Tensor,  # [..., n_gpus, 1]
):
    """Alpha compositing across all segments."""
    assert colors.ndim == accs.ndim == depths.ndim

    # sort by terminates along the GPU dimension
    indices = torch.argsort(terminates, dim=-2)
    accs_sorted = torch.gather(accs, -2, indices)

    # prefix transmittance = {1, (1-a1), (1-a1)*(1-a2), ...}
    trans_sorted = 1.0 - accs_sorted
    prefix_trans_sorted = torch.cumprod(
        torch.cat(
            [torch.ones_like(trans_sorted[..., :1, :]), trans_sorted[..., :-1, :]],
            dim=-2,
        ),
        dim=-2,
    )
    prefix_trans = torch.empty_like(prefix_trans_sorted)
    prefix_trans.scatter_(-2, indices, prefix_trans_sorted)

    # accumulate: c = c1 + (1-a1)*c2 + (1-a1)*(1-a2)*c3 + ...
    rendered_colors = (prefix_trans * colors).sum(dim=-2)
    rendered_depths = (prefix_trans * depths).sum(dim=-2)
    rendered_accs = (prefix_trans * accs).sum(dim=-2)
    return rendered_colors, rendered_accs, rendered_depths


def spatial_partition(
    points: Tensor,
    log2_n: int,
    scale: float = 1.0,
    eps: float = 1e-6,
    refine: bool = True,
    disable_axis: Optional[int] = None,
    uniform: bool = False,
) -> Tensor:
    """Split the space into 2^log2_n partitions with similar number of points in each partition.
    Args:
        points (Tensor): spatial points [N, 3]
        log2_n (int): log2 of the number of partitions.
        scale (float, optional): The scale factor for the aabb. Defaults to 1.0.
        refine: (bool, optional): Whether to refine the aabb to make them tight. Defaults to True.
        disable_axis: (int, optional): Disable the split on this axis. Defaults to None.
        uniform: (bool, optional): Uniform partitioning. Defaults to False.
    Returns:
        Tensor. The aabbs (x0, y0, z0, x1, y1, z1) for each partition. [2^log2_n, 6]
    """

    def _intersect(aabb1: Tensor, aabb2: Tensor) -> Tensor:
        return torch.cat([aabb1[:3].max(aabb2[:3]), aabb1[3:].min(aabb2[3:])])

    def _points_to_aabb(pts: Tensor) -> Tensor:
        minimam = pts.min(dim=0).values - eps
        maximam = pts.max(dim=0).values + eps
        center = 0.5 * (minimam + maximam)
        halfsize = 0.5 * (maximam - minimam)
        return torch.cat([center - scale * halfsize, center + scale * halfsize])

    def _aspect_ratio(aabb: Tensor) -> Tensor:
        return (aabb[3:] - aabb[:3]).min() / (aabb[3:] - aabb[:3]).max()

    def _split_axis(aabb: Tensor, dim: int) -> Tuple[Tensor, Tensor]:
        if uniform:
            split = ((aabb[:3] + aabb[3:]) * 0.5)[dim]
        else:
            mask = ((points >= aabb[:3]) & (points < aabb[3:])).all(dim=1)
            split = torch.quantile(points[mask, dim], 0.5, interpolation="linear")
        aabb1, aabb2 = aabb.clone(), aabb.clone()
        aabb1[3 + dim] = split  # max on this dim is set to `split`
        aabb2[dim] = split  # min on this dim is set to `split`
        if refine:
            assert not uniform, "Refinement is not supported for uniform partitioning."
            mask1 = ((points >= aabb1[:3]) & (points < aabb1[3:])).all(dim=1)
            aabb1_refine = _points_to_aabb(points[mask1])
            aabb1 = _intersect(aabb1, aabb1_refine)
            mask2 = ((points >= aabb2[:3]) & (points < aabb2[3:])).all(dim=1)
            aabb2_refine = _points_to_aabb(points[mask2])
            aabb2 = _intersect(aabb2, aabb2_refine)
        return aabb1, aabb2

    def _split(aabb: Tensor, lvl: int) -> Tuple[Tensor, ...]:
        if lvl == 0:
            return (aabb,)
        # Split along each axis.
        aabb1_a0, aabb2_a0 = _split_axis(aabb, 0)
        aabb1_a1, aabb2_a1 = _split_axis(aabb, 1)
        aabb1_a2, aabb2_a2 = _split_axis(aabb, 2)
        # Compute the aspect ratio of the AABBs.
        ratio_a0 = _aspect_ratio(aabb1_a0) * _aspect_ratio(aabb2_a0)
        ratio_a1 = _aspect_ratio(aabb1_a1) * _aspect_ratio(aabb2_a1)
        ratio_a2 = _aspect_ratio(aabb1_a2) * _aspect_ratio(aabb2_a2)
        if disable_axis == 0:
            ratio_a0 = -1
        elif disable_axis == 1:
            ratio_a1 = -1
        elif disable_axis == 2:
            ratio_a2 = -1
        else:
            assert disable_axis is None
        # Select the axis with the largest aspect ratio.
        if ratio_a0 >= ratio_a1 and ratio_a0 >= ratio_a2:
            aabb1, aabb2 = aabb1_a0, aabb2_a0
        elif ratio_a1 >= ratio_a0 and ratio_a1 >= ratio_a2:
            aabb1, aabb2 = aabb1_a1, aabb2_a1
        elif ratio_a2 >= ratio_a0 and ratio_a2 >= ratio_a1:
            aabb1, aabb2 = aabb1_a2, aabb2_a2
        if lvl == 1:
            return (aabb1, aabb2)
        else:
            return (*_split(aabb1, lvl - 1), *_split(aabb2, lvl - 1))

    # compute global aabb
    aabb = _points_to_aabb(points)
    return torch.stack(_split(aabb, log2_n), dim=0)
