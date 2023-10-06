Tests
===================================

Testing and verifying CUDA implementations
--------------------------------------------

The `tests/` folder provides automatic test scripts that are ran to verify that the CUDA implementations agree with native PyTorch ones.
They are also ran with any pull-requests into main branch on our github repository.

The tests include: 

.. code-block:: bash
    :caption: tests/

    ./test_bin_and_sort_gaussians.py
    ./test_cov2d_bounds.py
    ./test_cumulative_intersects.py
    ./test_get_tile_bin_edges.py
    ./test_map_gaussians.py
    ./test_project_gaussians
    ./test_rasterize_forward_kernel.py
    ./test_sh.py
