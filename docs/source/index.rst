gsplat
===================================

.. image:: imgs/training.gif
    :width: 800
    :alt: Example training image

Overview
--------

*gsplat* is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields" :cite:p:`kerbl3Dgaussians`.
This libary contains the neccessary components for efficient 3D to 2D projection, sorting, and alpha compositing of gaussians and their associated backward passes for inverse rendering.

Contributing
------------

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.


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
