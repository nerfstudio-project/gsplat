G-SHARP v0.2 Integration Proposal
==================================

:Status: Proposed (scaffolding landed on branch ``vnath_gsharp``)
:Source: ``holohub/applications/surgical_scene_recon/`` (G-SHARP v0.2)
:Destination: ``gsplat/`` (branch ``vnath_gsharp``)

Goals
-----

* Port all **training-time algorithmic** contributions of G-SHARP v0.2 natively
  into gsplat.
* Keep the deformable / 4D Gaussian machinery behind a clearly experimental
  ``gsplat/contrib/`` namespace.
* Ship test-first: every ported unit has positive + negative pytest cases.
* Provide a runnable end-to-end tutorial on the EndoNeRF sample.

Non-goals
---------

* No Holoscan / HoloHub imports anywhere.
* No Depth Anything V2 / MedSAM3 / VGGT code inside the ``gsplat/`` package.
  These are external pretrained nets; the tutorial only *references* them as
  optional upstream preprocessing.
* No compression / rate-distortion work (not present in G-SHARP v0.2).
* No camera-pose optimization (G-SHARP v0.2 doesn't do it either).

Component map
-------------

Core (stable) additions:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Component
     - New file
     - G-SHARP source
   * - Binocular disparity L1 loss
     - ``gsplat/losses_depth.py``
     - ``EndoRunner.compute_depth_loss``, ``training/gsplat_train.py`` 906–960
   * - Pearson depth loss (mono)
     - ``gsplat/losses_depth.py``
     - same
   * - Masked L1 / SSIM wrappers
     - ``gsplat/losses_depth.py``
     - ``training/gsplat_train.py`` 1059–1101
   * - Occlusion-TV regularizer
     - ``gsplat/regularizers.py``
     - ``compute_tv_loss_targeted``, 1925–2028
   * - Torch mask dilation
     - ``gsplat/regularizers.py``
     - OpenCV dilation replaced by max-pool
   * - Invisible-mask builder
     - ``gsplat/regularizers.py``
     - ``create_invisible_mask_from_paths``
   * - Multi-frame depth unprojection
     - ``gsplat/init_utils.py``
     - ``accumulate_multiframe_pointcloud``, 2036–2159
   * - KNN scale init
     - ``gsplat/init_utils.py``
     - ``create_splats_with_optimizers``, 1796–1906
   * - Two-stage scheduler
     - ``gsplat/training/schedulers.py``
     - ``EndoRunner._train_stage``, 1007–1050

Experimental contrib additions (``gsplat.contrib.dynamic``):

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Component
     - New file
     - G-SHARP source
   * - HexPlane field
     - ``gsplat/contrib/dynamic/hexplane.py``
     - ``training/scene/hexplane.py``
   * - Deform MLP + per-Gaussian deformation table
     - ``gsplat/contrib/dynamic/deformation.py``
     - ``training/scene/deformation.py``, ``gsplat_train.py`` 1376–1664
   * - Plane / time smoothness regularizers
     - ``gsplat/contrib/dynamic/regulation.py``
     - ``training/scene/regulation.py``, loop 1233–1272
   * - DynamicStrategy
     - ``gsplat/contrib/dynamic/strategy.py``
     - ``rasterize_splats``, 864–896

Examples / docs:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Deliverable
     - Path
   * - EndoNeRF/SCARED dataset loader
     - ``examples/datasets/endonerf.py``
   * - Dynamic-scene trainer recipe
     - ``examples/dynamic_trainer.py``
   * - Tutorial
     - ``docs/source/examples/dynamic_surgical.rst``
   * - Contrib API docs
     - ``docs/source/apis/contrib.rst``

Why this layout
---------------

* The core library stays pure splatting math. Depth losses / occlusion-TV /
  multi-frame init / two-stage schedule are generic and low-risk, so they live
  alongside existing ``losses.py``, ``rendering.py``, ``strategy/``.
* 4D / deformable pieces add new public surfaces (``HexPlane``, ``DeformNet``,
  ``DynamicStrategy``) that are still research-grade → ``gsplat/contrib/dynamic/``
  signals "ships with the wheel, API may change". Pattern follows the
  well-known ``torchvision.prototype`` convention.
* EndoNeRF/SCARED loader is domain-specific dataset glue → belongs with
  ``colmap.py`` / ``ncore.py`` under ``examples/datasets/``, not in the
  library.
* DA2 / MedSAM3 / VGGT **do not enter the tree**; the tutorial explains "get
  ``poses_bounds.npy``, ``masks/``, ``depth/`` some way" and points at the
  (unported) holohub folder for reference.

Test plan
---------

Every new module gets a ``tests/test_*.py`` with both positive and negative
cases. Tests follow the flat layout and pytest conventions already in
``gsplat/tests/`` (``conftest.py``, seed 42, CUDA-aware).

See the component × test-case matrix in
``planning/gsharp_gsplat_plan.md`` and the shareable HTML summary at
``planning/gsharp_gsplat_plan.html``.

Risks
-----

* **Deformation + ``rasterization()`` coupling.** ``gsplat.rasterization`` has
  no time axis. Solution: ``DynamicStrategy`` applies the deform-net to
  ``(means, quats, opacities)`` *before* ``rasterization(...)`` is called
  (same pattern as G-SHARP's ``rasterize_splats``).
* **CUDA-only tests.** HexPlane/DeformNet tests run on CPU for determinism;
  GPU smoke tests are marked CUDA-only.
* **EndoNeRF sample license.** Documented in the tutorial, not redistributed
  in-tree.
* **sklearn optional dep.** Pure-torch fallback for KNN init keeps the
  required-dep list unchanged.
