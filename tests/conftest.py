"""
Pytest configuration and shared fixtures for gsplat tests.

This file is automatically discovered by pytest and applies to all test files
in this directory and subdirectories.
"""

import pytest
import torch
import torch.distributed
import gc
import os

@pytest.fixture(autouse=True)
def setup_test_environment():
    """
    Autouse fixture that runs before every test to ensure:
    1. Deterministic random seed
    2. CUDA cache is cleared
    3. Garbage collection is performed

    This fixture automatically applies to all tests in this directory
    without needing to be explicitly requested.
    """

    seed = 42

    # Set seed based on test name for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    # Run garbage collection
    gc.collect()

    # Yield to run the test
    yield

    # Optional: cleanup after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="session")
def dist_init():
    """Initialize a single-process distributed group for testing distributed code paths.

    With world_size=1 the all-gather / all-to-all ops become identity operations,
    but the code path inside ``rasterization(distributed=True)`` is still exercised.
    """
    if not torch.cuda.is_available():
        yield
        return

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group(
            backend="nccl", world_size=1, rank=0
        )
        # Warm up the communicator required by batch_isend_irecv.
        _ = [None]
        torch.distributed.all_gather_object(_, 0)

    yield

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

