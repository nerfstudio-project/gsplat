Tests
===================================

.. currentmodule:: diff_rast

Testing and verifying CUDA implementations
--------------------------------------------

The `tests/` folder provides automatic test scripts that are ran to verify that the CUDA implementations agree with native PyTorch ones.
They are also ran with any pull-requests into main branch on our github repository.


.. code-block:: python
    :caption: tests/

    ./test_project_gaussians.py
    ./test_map_gaussians.py
    ./test_cumulative_intersects.py
    ./test_cov2d_bounds.py

