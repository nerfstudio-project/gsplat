gsplat
===================================

.. image:: assets/training.gif
    :width: 800
    :alt: Example training image

Overview
--------

*gsplat* is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields" :cite:p:`kerbl3Dgaussians`.
This libary contains the neccessary components for efficient 3D to 2D projection, sorting, and alpha compositing of gaussians and their associated backward passes for inverse rendering.

Contributing
------------

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.



+------------+-------+-------+-------+------------------+------------+
|            | PSNR  | SSIM  | LPIPS | Train Mem        | Train Time |
+============+=======+=======+=======+==================+============+
| inria-7k   | 27.23 | 0.829 | 0.204 | 7.7 GB           | 8m38s      |
+------------+-------+-------+-------+------------------+------------+
| gsplat-7k  | 27.21 | 0.831 | 0.202 | **4.3GB**        | **5m35s**  |
+------------+-------+-------+-------+------------------+------------+
| inria-30k  | 28.95 | 0.870 | 0.138 | 9.0 GB           | 48m29s     |
+------------+-------+-------+-------+------------------+------------+
| gsplat-30k | 28.95 | 0.870 | 0.135 | **5.7 GB**       | **35m49s** |
+------------+-------+-------+-------+------------------+------------+


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
