gsplat
===================================

.. image:: assets/training.gif
    :width: 800
    :alt: Example training image

Overview
--------

*gsplat* is an open-source library for CUDA accelerated differentiable rasterization of 
3D gaussians with python bindings. It is inspired by the SIGGRAPH paper "3D Gaussian Splatting for 
Real-Time Rendering of Radiance Fields" :cite:p:`kerbl3Dgaussians`, but we've made *gsplat* even 
faster, more memory efficient, and with a growing list of new features!

* *gsplat* is developed with efficiency in mind. Comparing to the `official implementation <https://github.com/graphdeco-inria/gaussian-splatting>`_, *gsplat* enables up to **4x less training memory footprint**, and up to **2x less training time** on Mip-NeRF 360 captures, and potential more on larger scenes. See :doc:`tests/eval` for details.

* *gsplat* is designed to **support extremely large scene rendering**, which is magnitudes faster than the official CUDA backend `diff-gaussian-rasterization <https://github.com/graphdeco-inria/diff-gaussian-rasterization>`_. See :doc:`examples/large_scale` for an example.

* *gsplat* offers many extra features, including **batch rasterization**,  **N-D feature rendering**, **depth rendering**, **sparse gradient** etc. See :doc:`apis/rasterization` for details.

* *gsplat* is equipped with the **latest and greatest** 3D Gaussian Splatting techniques, including `absgrad <https://ty424.github.io/AbsGS.github.io/>`_, `anti-aliasing <https://niujinshuchong.github.io/mip-splatting/>`_ etc. And more would come.


.. raw:: html
   
   <iframe width="784" height="441" src="https://www.youtube.com/embed/G4SmXplWIrY?si=t1zLvrpvPyjB2n52" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


Installation
------------

*gsplat* is available on PyPI, and can be installed with pip:

.. code-block:: bash

    pip install gsplat

Additionally, it can also be installed from source to get the latest features:

.. code-block:: bash

    pip install git+https://github.com/nerfstudio-project/gsplat

Contributing
------------

This repository was born from the curiosity of people on the Nerfstudio team trying to 
understand a new rendering technique. We welcome contributions of any kind and are open 
to feedback, bug-reports, and improvements to help expand the capabilities of this software.






Links
-----

.. toctree::    
   :glob:
   :maxdepth: 1
   :caption: Examples

   examples/*


.. toctree::    
   :glob:
   :maxdepth: 1
   :caption: Conventions

   conventions/*


.. toctree::    
   :glob:
   :maxdepth: 1
   :caption: Python API

   apis/*


.. toctree::    
   :glob:
   :maxdepth: 1
   :caption: Tests

   tests/*

.. toctree::    
   :glob:
   :maxdepth: 1
   :caption: Migration

   migration/*


Citations
-------------
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
