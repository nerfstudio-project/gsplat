# SPDX-License-Identifier: Apache-2.0
"""Scaffolding tests for the G-SHARP v0.2 additions to ``gsplat.losses`` (branch: vnath_gsharp).

Every test in this file is skipped until the corresponding implementation
lands. The skip markers document the intended positive / negative coverage.
"""

import pytest


@pytest.mark.skip(reason="vnath_gsharp: binocular_disparity_l1 not yet implemented")
def test_binocular_disparity_l1_identical_depths_is_zero():
    pass


@pytest.mark.skip(reason="vnath_gsharp: binocular_disparity_l1 not yet implemented")
def test_binocular_disparity_l1_gradient_flows():
    pass


@pytest.mark.skip(reason="vnath_gsharp: binocular_disparity_l1 not yet implemented")
def test_binocular_disparity_l1_mask_slices_correct():
    pass


@pytest.mark.skip(reason="vnath_gsharp: binocular_disparity_l1 not yet implemented")
def test_binocular_disparity_l1_shape_mismatch_raises():
    pass


@pytest.mark.skip(reason="vnath_gsharp: binocular_disparity_l1 not yet implemented")
def test_binocular_disparity_l1_all_masked_returns_zero_no_nan():
    pass


@pytest.mark.skip(reason="vnath_gsharp: binocular_disparity_l1 not yet implemented")
def test_binocular_disparity_l1_zero_depth_handled_via_eps():
    pass


@pytest.mark.skip(reason="vnath_gsharp: pearson_depth_loss not yet implemented")
def test_pearson_depth_loss_correlated_inputs_zero():
    pass


@pytest.mark.skip(reason="vnath_gsharp: pearson_depth_loss not yet implemented")
def test_pearson_depth_loss_anticorrelated_inputs_two():
    pass


@pytest.mark.skip(reason="vnath_gsharp: masked_l1 not yet implemented")
def test_masked_l1_ignores_masked_out_pixels():
    pass


@pytest.mark.skip(reason="vnath_gsharp: masked_l1 not yet implemented")
def test_masked_l1_empty_mask_returns_zero_no_nan():
    pass
