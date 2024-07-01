import torch
import torch.distributed
import torch.distributed.nn.functional as dist_func

from gsplat.core import cli


def main(local_rank: int, world_rank, world_size: int, _):
    world_rank == torch.distributed.get_rank()
    world_size == torch.distributed.get_world_size()

    device = torch.device("cuda", local_rank)
    input = (
        torch.arange(world_size, dtype=torch.float32, device=device)
        + world_rank * world_size
    )
    input.requires_grad = True

    # input_l = list(input.chunk(world_size))
    # output = list(
    #     torch.empty([world_size], dtype=torch.float32, device=device).chunk(world_size)
    # )
    # dist_func.all_to_all(output, input_l)

    output = torch.empty([world_size], dtype=torch.float32, device=device)
    dist_func.all_to_all_single(output, input, [1] * world_size, [1] * world_size)

    sum(output).sum().backward()
    print("rank: ", world_rank, "grad: ", input.grad)


if __name__ == "__main__":
    cli(main, None, verbose=True)
