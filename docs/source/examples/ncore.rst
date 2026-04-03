Train on an NCore v4 Capture
========================================

.. currentmodule:: gsplat

The :code:`examples/simple_trainer.py` script supports training a
`3D Gaussian Splatting <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>`_
model on an `NCore v4 <https://nvidia.github.io/ncore/>`_ capture.
Point ``--data-dir`` at the sequence meta-JSON file and set ``--data-type ncore``:

.. code-block:: bash

    cd examples
    python simple_trainer.py default \
        --data-type ncore \
        --data-dir /path/to/sequence.json \
        --data-factor 1 \
        --result-dir results/my_scene \
        --init-type lidar

For datasets with FTheta cameras (auto-detected from sequence metadata):

.. code-block:: bash

    cd examples
    python simple_trainer.py mcmc \
        --data-type ncore \
        --data-dir /path/to/sequence.json \
        --data-factor 1 \
        --result-dir results/my_scene \
        --init-type lidar \
        --with-ut \
        --with-eval3d

.. note::

   The camera model (pinhole, fisheye, or FTheta) is read automatically from the NCore
   sequence metadata. Passing ``--camera-model`` has no effect when ``--data-type ncore``
   is used.

.. note::

   Training with ``--data-factor`` values other than ``1`` requires that all
   camera resolutions are divisible by the factor.

.. note::

   World-space normalization (PCA + scale) is applied by default
   (``--normalize-world-space`` is ``True``). For outdoor or large-scale captures
   where preserving real-world scale matters, pass ``--no-normalize-world-space``.

.. note::

   When a sequence contains **more than one camera or lidar sensor**, you must
   explicitly specify which sensor(s) to use via ``--ncore-camera-ids`` and/or
   ``--ncore-lidar-ids``.  If omitted, the parser auto-detects sensors but will
   raise an error when multiple sensors are present to avoid ambiguity
   (e.g. when multiple downscaled variants of the same sensor exist).

   .. code-block:: bash

       cd examples
       python simple_trainer.py default \
           --data-type ncore \
           --data-dir /path/to/sequence.json \
           --ncore-camera-ids camera_front \
           --ncore-lidar-ids lidar_top \
           ...
