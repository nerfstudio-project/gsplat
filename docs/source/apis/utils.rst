Utils
===================================
In addition to the main projection and rasterization functions, a few CUDA kernel and helper functions are exposed to python with the following bindings:


.. currentmodule:: gsplat

.. autofunction:: bin_an_sort_gaussians

.. autofunction:: compute_cov2d_bounds

.. autofunction:: get_tile_bin_edges

.. autofunction:: spherical_harmonics

.. autofunction:: map_gaussian_to_intersects

.. autofunction:: compute_cumulative_intersects