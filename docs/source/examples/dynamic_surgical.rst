Dynamic Surgical Scene Reconstruction (G-SHARP port, experimental)
===================================================================

.. note::

   This pipeline is **experimental**. The G-SHARP port lives under
   ``gsplat.contrib.dynamic`` (HexPlane field, deformation network,
   :class:`~gsplat.contrib.dynamic.DynamicStrategy`) plus surgical-scene
   helpers in ``gsplat.losses``, ``gsplat.regularizers``,
   ``gsplat.init_utils``, ``gsplat.training``. APIs may change. See
   :doc:`../proposals/gsharp_v0_2_port` for the design background.

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

The tutorial targets the EndoNeRF sample ("pulling"). SCARED support is
**not yet implemented** in this MR; ``examples/datasets/`` only ships an
EndoNeRF parser and a stub for SCARED.

gsplat itself does **not** ship code for estimating depth, tool masks, or
camera poses. If you don't already have these, you can generate them with
any standard tool; the G-SHARP v0.2 holohub application uses
Depth Anything V2, MedSAM3, and VGGT-1B as an optional upstream preprocessing
stack — see ``holohub/applications/surgical_scene_recon`` for a reference
implementation.

Getting the data
~~~~~~~~~~~~~~~~

The EndoNeRF "pulling" sequence is hosted on the upstream
`med-air/EndoNeRF <https://github.com/med-air/EndoNeRF>`_ project's
Google Drive folder
(`direct link <https://drive.google.com/drive/folders/1zTcX80c1yrbntY9c6-EK2W2UVESVEug8?usp=sharing>`_).
Download the ``pulling_soft_tissues`` directory and rename to
``data/EndoNeRF/pulling`` (or wherever you point ``--data_dir`` at).

Expected on-disk layout::

    <scene>/
      poses_bounds.npy
      images/ 000000.png 000001.png ...
      depth/  000000.png 000001.png ...
      masks/  000000.png 000001.png ...

Quickstart
----------

.. code-block:: bash

   python examples/dynamic_surgical_trainer.py \
       --data_dir path/to/endonerf_pulling \
       --output_dir output/dynamic_surgical \
       --coarse_steps 200 --fine_steps 2500 \
       --init_max_points 50000 \
       --depth_mode binocular \
       --render_gif_after_train

The trainer auto-derives the HexPlane AABB from the init point cloud
(see :func:`examples.dynamic_surgical_trainer.train`); ``--hex_bounds``
remains a lower bound. Run ``python examples/dynamic_surgical_trainer.py
--help`` for the complete flag list (it's a tyro dataclass surface on
:class:`Config`).

What lands next
---------------

Implementation will be driven test-first. Progress is tracked in
``planning/dev_updates/`` at the repo's parent workspace.
