import pytest
import torch

device = torch.device("cuda:0")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_build_H():
    from gsplat import ...