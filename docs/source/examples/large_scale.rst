Render a Large Scene 
========================================

.. currentmodule:: gsplat

`gsplat` is designed with efficiency in mind so it's very suitable to render a large scene.
For example here we mimic a large scene by replicating the Garden scene into a 9x9 grid, 
which results 30M Gaussians in total while `gsplat` still allows real-time rendering for it.

With `gsplat` as CUDA Backend:

.. raw:: html

    <iframe width="784" height="441" src="https://www.youtube.com/embed/rXyJwL7uJFQ?si=kQ10HcASlvhztTUA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

With `diff-gaussian-rasterization` as CUDA Backend:

.. raw:: html

    <iframe width="784" height="441" src="https://www.youtube.com/embed/kk3G3P68rkc?si=JJplV2ZY3PxE_5k7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Note: Similar to the `nerfstudio <https://docs.nerf.studio/>`_ viewer, our viewer automatically 
switch to low resolution if the rendering is slow.

The code for this example can be found under `examples/`:

.. code-block:: bash

    # First train a 3DGS model
    python simple_trainer.py \
        --data_dir data/360_v2/garden/ --data_factor 4 \
        --result_dir ./results/garden

    # View it in a viewer with gsplat
    python simple_viewer.py --scene_grid 5 --ckpt results/garden/ckpts/ckpt_6999.pt --backend gsplat

    # Or, view it with inria's backend (requires to insteall `diff-gaussian-rasterization`)
    python simple_viewer.py --scene_grid 5 --ckpt results/garden/ckpts/ckpt_6999.pt --backend inria

    # Warning: a large `--scene_grid` might blow up your GPU memory.

