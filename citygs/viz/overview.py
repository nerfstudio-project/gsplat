"""Top-down partition overview (matplotlib, contracted space).

One picture answering: where are the blocks, where did the Gaussians go,
where are the cameras. Written by the partition stage as partition.png.
"""

from typing import Optional

import numpy as np

from ..scene.manifest import SceneManifest


def save_partition_overview(
    manifest: SceneManifest,
    gaussians_contracted: np.ndarray,  # [N, 3]
    gaussian_block_ids: np.ndarray,  # [N]
    cameras_contracted: np.ndarray,  # [C, 3]
    out_path: str,
    max_points: int = 200_000,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(gaussians_contracted)
    if n > max_points:
        sel = np.random.default_rng(0).choice(n, max_points, replace=False)
        pts = gaussians_contracted[sel]
        ids = gaussian_block_ids[sel]
    else:
        pts, ids = gaussians_contracted, gaussian_block_ids

    num_blocks = max(len(manifest.blocks), 1)
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=[cmap(i % 20) for i in ids],
        s=0.3,
        alpha=0.4,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        cameras_contracted[:, 0],
        cameras_contracted[:, 1],
        c="black",
        s=4,
        marker="^",
        label=f"cameras ({len(cameras_contracted)})",
    )
    for block in manifest.blocks:
        x0, y0 = block.bmin[0], block.bmin[1]
        w, h = block.bmax[0] - x0, block.bmax[1] - y0
        ax.add_patch(
            Rectangle((x0, y0), w, h, fill=False, edgecolor="black", linewidth=0.8)
        )
        cx = np.clip((x0 + block.bmax[0]) / 2, -1.6, 1.6)
        cy = np.clip((y0 + block.bmax[1]) / 2, -1.6, 1.6)
        ax.text(
            cx,
            cy,
            f"{block.block_id}\n{block.num_gaussians / 1e6:.2f}M / "
            f"{len(block.cameras)}c",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7, lw=0),
        )
    # Foreground box [-1, 1]^2.
    ax.add_patch(
        Rectangle((-1, -1), 2, 2, fill=False, edgecolor="red", ls="--", lw=1.2)
    )
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.set_aspect("equal")
    ax.set_title(
        f"partition overview (contracted space) — {num_blocks} blocks, "
        f"{n / 1e6:.1f}M gaussians"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
