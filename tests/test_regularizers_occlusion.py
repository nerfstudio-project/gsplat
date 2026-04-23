# SPDX-License-Identifier: Apache-2.0
"""Scaffolding tests for gsplat.regularizers (branch: vnath_gsharp)."""

import pytest


@pytest.mark.skip(reason="vnath_gsharp: compute_tv_loss_targeted not yet implemented")
def test_tv_targeted_constant_image_is_zero():
    pass


@pytest.mark.skip(reason="vnath_gsharp: compute_tv_loss_targeted not yet implemented")
def test_tv_targeted_step_edge_matches_analytic_value():
    pass


@pytest.mark.skip(reason="vnath_gsharp: compute_tv_loss_targeted not yet implemented")
def test_tv_targeted_empty_mask_returns_zero_no_div_by_zero():
    pass


@pytest.mark.skip(reason="vnath_gsharp: compute_tv_loss_targeted not yet implemented")
def test_tv_targeted_non_binary_mask_rejected():
    pass


@pytest.mark.skip(reason="vnath_gsharp: dilate_mask not yet implemented")
def test_dilate_mask_kernel_one_is_identity():
    pass


@pytest.mark.skip(reason="vnath_gsharp: dilate_mask not yet implemented")
def test_dilate_mask_kernel_three_grows_by_one_pixel():
    pass


@pytest.mark.skip(reason="vnath_gsharp: create_invisible_mask not yet implemented")
def test_create_invisible_mask_union_of_inputs():
    pass
