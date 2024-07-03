import os
from typing import Any, Callable, List, Optional, Tuple

import torch


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
