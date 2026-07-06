import numpy as np

from citygs.partition.partition import (
    CONTRACTED_LIMIT,
    assign_gaussians_to_blocks,
    block_mask,
    make_grid_blocks,
)


def test_grid_blocks_tile_contracted_cube():
    blocks = make_grid_blocks((3, 2))
    assert len(blocks) == 6
    # Outermost cells reach the contracted limit.
    xs = [b.bmin[0] for b in blocks] + [b.bmax[0] for b in blocks]
    ys = [b.bmin[1] for b in blocks] + [b.bmax[1] for b in blocks]
    assert min(xs) == -CONTRACTED_LIMIT and max(xs) == CONTRACTED_LIMIT
    assert min(ys) == -CONTRACTED_LIMIT and max(ys) == CONTRACTED_LIMIT
    # Interior cut lines are on the foreground grid.
    interior = sorted({b.bmax[0] for b in blocks} - {CONTRACTED_LIMIT})
    np.testing.assert_allclose(interior, [-1 / 3, 1 / 3], atol=1e-12)


def test_gaussian_assignment_is_a_partition():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.999, 1.999, size=(5000, 3))
    blocks = make_grid_blocks((4, 3))
    ids = assign_gaussians_to_blocks(pts, blocks)
    assert ids.min() >= 0 and ids.max() < 12
    # Each point in exactly one block; counts recorded per block.
    assert sum(b.num_gaussians for b in blocks) == len(pts)
    for b in blocks:
        assert (ids == b.block_id).sum() == b.num_gaussians


def test_boundary_points_unique_assignment():
    blocks = make_grid_blocks((2, 1))
    # Points exactly on the internal cut x=0 belong to the upper cell.
    pts = np.array([[0.0, 0.0, 0.0], [-1e-9, 0.0, 0.0]])
    ids = assign_gaussians_to_blocks(pts, blocks)
    assert ids[0] == 1 and ids[1] == 0


def test_block_mask_margin_overlaps():
    blocks = make_grid_blocks((2, 1))
    p = np.array([[0.05, 0.0, 0.0]])  # inside block 1, near the border
    assert not block_mask(p, blocks[0]).any()
    assert block_mask(p, blocks[0], margin=0.1).all()
    assert block_mask(p, blocks[1]).all()
