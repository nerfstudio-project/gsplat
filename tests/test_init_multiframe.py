# SPDX-License-Identifier: Apache-2.0
"""Scaffolding tests for gsplat.init_utils (branch: vnath_gsharp)."""

import pytest


@pytest.mark.skip(reason="vnath_gsharp: multi_frame_depth_unprojection not yet implemented")
def test_multiframe_unprojection_recovers_synthetic_cube():
    pass


@pytest.mark.skip(reason="vnath_gsharp: multi_frame_depth_unprojection not yet implemented")
def test_multiframe_unprojection_rgb_matches_source_pixels():
    pass


@pytest.mark.skip(reason="vnath_gsharp: multi_frame_depth_unprojection not yet implemented")
def test_multiframe_unprojection_frame_count_mismatch_raises():
    pass


@pytest.mark.skip(reason="vnath_gsharp: multi_frame_depth_unprojection not yet implemented")
def test_multiframe_unprojection_all_masked_returns_empty_point_set():
    pass


@pytest.mark.skip(reason="vnath_gsharp: knn_scale_init not yet implemented")
def test_knn_scale_init_matches_sklearn_reference():
    pass
