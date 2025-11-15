"""Tests for mathematical operations in gsplat.cuda._math module."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_quat_to_rotmat():
    """Test _quat_to_rotmat converts quaternions correctly."""
    from gsplat.cuda._math import _quat_to_rotmat

    # Test 1: Identity quaternion → identity matrix
    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    R = _quat_to_rotmat(q_identity)
    I = torch.eye(3, device=device).unsqueeze(0)
    torch.testing.assert_close(R, I, rtol=1e-6, atol=1e-6)

    # Test 2: 180-degree rotation around X-axis
    # q = (0, 1, 0, 0)
    q_x180 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    R = _quat_to_rotmat(q_x180)
    R_expected = torch.tensor([[[1.0, 0.0, 0.0],
                                 [0.0, -1.0, 0.0],
                                 [0.0, 0.0, -1.0]]], device=device)
    torch.testing.assert_close(R, R_expected, rtol=1e-5, atol=1e-5)

    # Test 3: random
    quats_large = torch.randn(1000, 4, device=device)
    quats_large = F.normalize(quats_large, p=2, dim=-1)
    R_large = _quat_to_rotmat(quats_large)

    # Verify all are valid rotation matrices
    det = torch.linalg.det(R_large)
    torch.testing.assert_close(det, torch.ones_like(det), rtol=1e-5, atol=1e-5)


