"""Grid partition in contracted space.

Blocks are cells of an (nx, ny) grid over the contracted foreground
[-1, 1]^2 in x/y. Outermost cells are extended to +-2 so the cells tile
the whole contracted cube (background belongs to its nearest border
block). z is never split: city scenes are flat, and vertical cuts create
boundary artifacts through every facade.

The number of blocks is a property of the scene, decoupled from the GPU
count — the finetune scheduler packs blocks onto however many GPUs exist.
"""

from typing import List, Tuple

import numpy as np

from ..scene.manifest import BlockMeta

# Contracted space is the open cube (-2, 2)^3.
CONTRACTED_LIMIT = 2.0


def make_grid_blocks(grid: Tuple[int, int]) -> List[BlockMeta]:
    nx, ny = grid
    assert nx >= 1 and ny >= 1
    xs = np.linspace(-1.0, 1.0, nx + 1)
    ys = np.linspace(-1.0, 1.0, ny + 1)
    blocks = []
    for j in range(ny):
        for i in range(nx):
            x0 = -CONTRACTED_LIMIT if i == 0 else xs[i]
            x1 = CONTRACTED_LIMIT if i == nx - 1 else xs[i + 1]
            y0 = -CONTRACTED_LIMIT if j == 0 else ys[j]
            y1 = CONTRACTED_LIMIT if j == ny - 1 else ys[j + 1]
            blocks.append(
                BlockMeta(
                    block_id=len(blocks),
                    grid_index=(i, j),
                    bmin=[float(x0), float(y0), -CONTRACTED_LIMIT],
                    bmax=[float(x1), float(y1), CONTRACTED_LIMIT],
                )
            )
    return blocks


def block_mask(
    contracted: np.ndarray, block: BlockMeta, margin: float = 0.0
) -> np.ndarray:
    """Mask of contracted points inside a block cell (optionally expanded).

    Cells are half-open [bmin, bmax) except at the +2 boundary, so with
    margin=0 every point belongs to exactly one block.
    """
    bmin = np.asarray(block.bmin) - margin
    bmax = np.asarray(block.bmax) + margin
    upper_closed = np.asarray(block.bmax) >= CONTRACTED_LIMIT
    below = np.where(upper_closed | (margin > 0), contracted <= bmax, contracted < bmax)
    return ((contracted >= bmin) & below).all(axis=-1)


def assign_gaussians_to_blocks(
    contracted: np.ndarray, blocks: List[BlockMeta]
) -> np.ndarray:
    """Assign each contracted point to its (unique) block. Returns int32
    block ids, and fills ``block.num_gaussians``."""
    ids = np.full(len(contracted), -1, dtype=np.int32)
    for block in blocks:
        mask = block_mask(contracted, block)
        ids[mask] = block.block_id
        block.num_gaussians = int(mask.sum())
    assert (ids >= 0).all(), "grid cells must tile the contracted cube"
    return ids
