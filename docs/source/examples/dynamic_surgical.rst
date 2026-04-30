Dynamic Surgical Scene Reconstruction (G-SHARP port, experimental)
===================================================================

.. note::

   This tutorial is **scaffolding only** on branch ``vnath_gsharp``. The
   underlying modules (the G-SHARP additions in ``gsplat.losses``, ``gsplat.regularizers``,
   ``gsplat.init_utils``, ``gsplat.training``, ``gsplat.contrib.dynamic``) are
   stubs pending implementation. See the proposal at
   :doc:`../proposals/gsharp_v0_2_port` for the full plan.

Overview
--------

This example reconstructs a dynamic surgical scene (deforming tissue plus
moving tools) on top of the core ``gsplat.rasterization`` pipeline. It ports
the training-time components of G-SHARP v0.2 (see the holohub application
``surgical_scene_recon``):

* Depth supervision — binocular disparity L1 or monocular Pearson
* Tool / dynamic-object masking of RGB and depth losses
* Invisible-region targeted TV regularizer
* Multi-frame depth unprojection for SfM-free point cloud init
* Two-stage schedule: coarse static → fine dynamic
* 4D Gaussians via ``gsplat.contrib.dynamic`` (HexPlane + MLP deformation)

Data
----

The tutorial targets the EndoNeRF sample ("pulling") and the SCARED dataset.
gsplat itself does **not** ship code for estimating depth, tool masks, or
camera poses. If you don't already have these, you can generate them with
any standard tool; the G-SHARP v0.2 holohub application uses
Depth Anything V2, MedSAM3, and VGGT-1B as an optional upstream preprocessing
stack — see ``holohub/applications/surgical_scene_recon`` for a reference
implementation.

Expected on-disk layout::

    <scene>/
      poses_bounds.npy
      images/ 000000.png 000001.png ...
      depth/  000000.png 000001.png ...
      masks/  000000.png 000001.png ...

Quickstart
----------

.. code-block:: bash

   # (once the vnath_gsharp branch lands implementation)
   python examples/dynamic_surgical_trainer.py \
       --data_dir path/to/endonerf_pulling \
       --dataset_type endonerf \
       --depth_mode binocular \
       --strategy dynamic

What lands next
---------------

Implementation will be driven test-first. Progress is tracked in
``planning/dev_updates/`` at the repo's parent workspace.
