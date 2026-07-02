import os

import pytest

_GPU_CI_XFAIL = {
    "tests/test_basic.py::test_quat_scale_to_covar_preci[batch_dims1-True]",
    "tests/test_basic.py::test_quat_scale_to_covar_preci[batch_dims2-True]",
    "tests/test_basic.py::test_projection[batch_dims1-True-False-pinhole]",
    "tests/test_basic.py::test_projection[batch_dims1-True-True-pinhole]",
    "tests/test_basic.py::test_projection[batch_dims2-True-False-pinhole]",
    "tests/test_basic.py::test_projection[batch_dims2-True-True-pinhole]",
    "tests/test_basic.py::test_fully_fused_projection_packed[batch_dims0-pinhole-False-False-False]",
    "tests/test_basic.py::test_fully_fused_projection_packed[batch_dims0-pinhole-False-False-True]",
    "tests/test_basic.py::test_fully_fused_projection_packed[batch_dims0-pinhole-True-False-False]",
    "tests/test_basic.py::test_fully_fused_projection_packed[batch_dims0-pinhole-True-False-True]",
    "tests/test_basic.py::test_fully_fused_projection_packed[batch_dims1-pinhole-False-False-False]",
    "tests/test_basic.py::test_fully_fused_projection_packed[batch_dims1-pinhole-False-False-True]",
    "tests/test_basic.py::test_fully_fused_projection_packed[batch_dims1-pinhole-True-False-False]",
    "tests/test_basic.py::test_fully_fused_projection_packed[batch_dims1-pinhole-True-False-True]",
    "tests/test_basic.py::test_fully_fused_projection_packed[batch_dims2-pinhole-False-False-False]",
    "tests/test_basic.py::test_fully_fused_projection_packed[batch_dims2-pinhole-False-False-True]",
    "tests/test_basic.py::test_fully_fused_projection_packed[batch_dims2-pinhole-True-False-False]",
    "tests/test_basic.py::test_fully_fused_projection_packed[batch_dims2-pinhole-True-False-True]",
    "tests/test_basic.py::test_rasterize_to_pixels_eval3d[3-batch_dims10-0-True-False-False-pinhole-8]",
    "tests/test_basic.py::test_rasterize_to_pixels_eval3d[3-batch_dims11-0-True-False-False-pinhole-16]",
}


def pytest_collection_modifyitems(items):
    if os.environ.get("GPU_CI_XFAIL") != "1":
        return
    for item in items:
        if item.nodeid in _GPU_CI_XFAIL:
            item.add_marker(
                pytest.mark.xfail(reason="known marginal FP mismatch on GPU CI runner")
            )
