from typing import List, Tuple

# mypy: allow-untyped-defs
import torch
import torch.distributed
from torch import Tensor
from torch.distributed.nn.functional import all_gather, all_to_all

from gsplat.core import cli, profiler, timeit


@timeit()
def _dist_gather_camera(
    world_rank: int, world_size: int, viewmats: Tensor, Ks: Tensor
) -> Tuple[Tensor, Tensor]:
    # Gather cameras from all ranks. We assume the number of cameras 
    # is the same across all ranks.
    if world_size == 1:
        assert world_rank == 0, "world_size is 1, but world_rank is not 0."
    else:
        device = viewmats.device
        C = len(viewmats)

        # [TODO] `all_gather` is not differentiable w.r.t. viewmats and Ks
        # {'_dist_gather_camera': 1.4657024070620537}
        # out_tensor_list = [
        #     torch.empty((C, 4, 4), device=device) for _ in range(world_size)
        # ]
        # torch.distributed.all_gather(out_tensor_list, viewmats.contiguous())
        # viewmats = torch.cat(out_tensor_list, dim=0)

        # out_tensor_list = [
        #     torch.empty((C, 3, 3), device=device) for _ in range(world_size)
        # ]
        # torch.distributed.all_gather(out_tensor_list, Ks.contiguous())
        # Ks = torch.cat(out_tensor_list, dim=0)

        out_tensor_list = [
            torch.empty((C, 4 * 4 + 3 * 3), device=device) for _ in range(world_size)
        ]
        data = torch.cat([viewmats.view(C, -1), Ks.view(C, -1)], dim=1)
        torch.distributed.all_gather(out_tensor_list, data.contiguous())
        data = torch.cat(out_tensor_list, dim=0)

        viewmats = data[:, :4 * 4].view(-1, 4, 4)
        Ks = data[:, 4 * 4:].view(-1, 3, 3)

        # out_tensor_list = [
        #     torch.empty((C, 3, 3), device=device) for _ in range(world_size)
        # ]
        # torch.distributed.all_gather(out_tensor_list, Ks.contiguous())
        # Ks = torch.cat(out_tensor_list, dim=0)


    return viewmats, Ks

@timeit()
def _dist_gather_int32(
    world_rank: int, world_size: int, value: int
) -> List[int]:
    collected = [None] * world_size
    torch.distributed.all_gather_object(collected, value)
    return collected

@timeit()
def _dist_gather_int32_2(
    world_rank: int, world_size: int, value: int
) -> List[int]:
    device = torch.device("cuda", world_rank)
    value_tensor = torch.tensor(value, dtype=torch.int, device=device)

    collected = torch.empty(world_size, dtype=torch.int, device=device)
    torch.distributed.all_gather_into_tensor(collected, value_tensor)
    return collected.tolist()

@timeit()
def _dist_all_gather_non_diff(
    world_rank: int, world_size: int, tensor_list: List[Tensor]
) -> List[Tensor]:
    out_tensor_list = []
    for tensor in tensor_list:
        collected = [
            torch.empty_like(tensor) for _ in range(world_size)
        ]
        torch.distributed.all_gather(collected, tensor)
        collected = torch.cat(collected, dim=0)
        out_tensor_list.append(collected)
    return out_tensor_list
    
@timeit()
def _dist_all_gather_non_diff_2(
    world_rank: int, world_size: int, tensor_list: List[Tensor]
) -> List[Tensor]:
    N = len(tensor_list[0])

    data = torch.cat([t.reshape(N, -1) for t in tensor_list], dim=-1)
    sizes = [t.numel() // N for t in tensor_list]
    collected = [torch.empty_like(data) for _ in range(world_size)]
    torch.distributed.all_gather(collected, data)
    collected = torch.cat(collected, dim=0)
    
    out_tensor_tuple = torch.split(collected, sizes, dim=-1)
    out_tensor_list = []
    for i, (out_tensor, tensor) in enumerate(zip(out_tensor_tuple, tensor_list)):
        out_tensor = out_tensor.view(N * world_size, *tensor.shape[1:])
        out_tensor_list.append(out_tensor)
    return out_tensor_list
    
@timeit()
def _dist_all_gather(
    world_rank: int, world_size: int, tensor_list: List[Tensor]
) -> List[Tensor]:
    out_tensor_list = []
    for tensor in tensor_list:
        collected = all_gather(tensor)
        collected = torch.cat(collected, dim=0)
        out_tensor_list.append(collected)
    return out_tensor_list
    
@timeit()
def _dist_all_gather_2(
    world_rank: int, world_size: int, tensor_list: List[Tensor]
) -> List[Tensor]:
    N = len(tensor_list[0])

    data = torch.cat([t.reshape(N, -1) for t in tensor_list], dim=-1)
    sizes = [t.numel() // N for t in tensor_list]

    collected = all_gather(data)
    collected = torch.cat(collected, dim=0)
    
    out_tensor_tuple = torch.split(collected, sizes, dim=-1)
    out_tensor_list = []
    for out_tensor, tensor in zip(out_tensor_tuple, tensor_list):
        out_tensor = out_tensor.view(N * world_size, *tensor.shape[1:])
        out_tensor_list.append(out_tensor)
    return out_tensor_list

    
@timeit()
def _dist_all_to_all_2(
    world_rank: int, world_size: int, tensor_list: List[Tensor], 
    input_cnts: List[int], output_cnts: List[int]
) -> List[Tensor]:
    N = len(tensor_list[0])

    data = torch.cat([t.reshape(N, -1) for t in tensor_list], dim=-1)
    sizes = [t.numel() // N for t in tensor_list]
    
    collected = [
        torch.empty((l, *data.shape[1:]), dtype=data.dtype, device=data.device) 
        for l in output_cnts
    ]
    all_to_all(collected, data.split(input_cnts, dim=0))
    collected = torch.cat(collected, dim=0)
    
    out_tensor_tuple = torch.split(collected, sizes, dim=-1)
    out_tensor_list = []
    for out_tensor, tensor in zip(out_tensor_tuple, tensor_list):
        out_tensor = out_tensor.view(-1, *tensor.shape[1:])
        out_tensor_list.append(out_tensor)
    return out_tensor_list


def main(local_rank: int, world_rank, world_size: int, _):
    world_rank == torch.distributed.get_rank()
    world_size == torch.distributed.get_world_size()

    device = torch.device("cuda", local_rank)

    C = 10000
    viewmats = torch.rand((C, 4, 4), device=device)
    Ks = torch.rand((C, 3, 3), device=device)

    # for _ in range(100):
    #     _ = _dist_gather_int32_2(world_rank, world_size, 3)
    # for _ in range(100):
    #     _ = _dist_gather_int32(world_rank, world_size, 3)
    # {'_dist_gather_integers': 0.16876974957995117, '_dist_gather_integers2': 0.028658458031713963}

    # for _ in range(100):
    #     _ = _dist_gather_camera(world_rank, world_size, viewmats, Ks)

    tensor1 = torch.rand((C, 4, 4), device=device, requires_grad=True)
    tensor2 = torch.rand((C, 3, 3), device=device, requires_grad=False)

    # for _ in range(100):
    #     _ = _dist_all_gather_non_diff(world_rank, world_size, [tensor1, tensor2])
    # for _ in range(100):
    #     _ = _dist_all_gather_non_diff_2(world_rank, world_size, [tensor1, tensor2])

    for _ in range(100):
        a = _dist_all_gather(world_rank, world_size, [tensor1, tensor2])
    with timeit("_dist_all_gather backward"):
        for _ in range(1):
            sum([t.sum() for t in a]).backward(retain_graph=True)

    # for _ in range(100):
    #     a = _dist_all_gather_2(world_rank, world_size, [tensor1, tensor2])
    # with timeit("_dist_all_gather_2 backward"):
    #     for _ in range(1):
    #         sum([t.sum() for t in a]).backward(retain_graph=True)    
    print(profiler)

if __name__ == "__main__":
    """
    TIMEIT=1 CUDA_VISIBLE_DEVICES=4,5,6,7 CUDA_LAUNCH_BLOCKING=1 python tests/dist2.py 
    """
    cli(main, None, verbose=True)
