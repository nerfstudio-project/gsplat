diff_rast
===================================

.. image:: imgs/training.gif
    :width: 800
    :alt: Example training image

Overview
-------------

*diff_rast* is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields" :cite:p:`kerbl3Dgaussians`.
This libary contains the neccessary components for efficient 3D to 2D projection, sorting, and alpha compositing of gaussians and their associated backward passes for inverse rendering.

Links
------

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
   :caption: Cuda Backend

   _C/*


.. toctree::    
   :glob:
   :maxdepth: 1
   :caption: Tests

   tests/*

Citations
-------------
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
