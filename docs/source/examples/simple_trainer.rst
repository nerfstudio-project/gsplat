Simple Trainer
===================================

.. currentmodule:: gsplat

Training on an image
-----------------------------------
The `examples/simple_trainer.py` script allows you to test the basic projection and rasterization operations with randomly initialized Gaussians
and their projection onto a single image. This allows you to overfit Gaussians on a single view.

Run the script with:

.. code-block:: bash
    :caption: simple_trainer.py

    python examples/simple_trainer.py --height 256 --width 256 --num_points 2000 --save_imgs

to get a result similar to the one below:

.. image:: ../imgs/square.gif
    :alt: Gaussians overfit on a single image
    :width: 256

You can also provide a path to your own custom image file using the ``--img_path`` flag:

.. code-block:: bash

    python examples/simple_trainer.py --img_path PATH_TO_IMG --save_imgs
