CUDA Conventions
===================================

.. currentmodule:: diff_rast

Here we explain some conventions used for the implementation of the CUDA backend.

Kernel Launch Conventions
-------------------------

The backend CUDA code consists of several kernel functions that run in parallel for either arbitrary number of gaussians or for arbitrary image dimensions.
Here we explain some of the conventions used for launching these kernel functions using the ``<<<gridDim, blockDim>>>`` notation.


1D Grids
^^^^^^^^

Kernel functions that depend on the number of input gaussians are organized into one dimensional CUDA grids consisting of 

.. code-block::

    const int gridDim = (num_gaussian_operations + NUM_THREADS - 1) / NUM_THREADS;
    
1-D blocks containing ``const int blockDim = NUM_THREADS;`` threads each. An example of a kernel function that depends on the number of gaussians (here denoted by the ``num_gaussian_operations`` variable)
is the main `project_gaussians` kernel that projects an arbitrary number of 3D gaussians into 2D.


2D Grids
^^^^^^^^

Kernel functions that depend on the rendered output image size are organized into two dimensional CUDA grids. The shape of the grid is determined by 

.. code-block:: 
    
    const dim3 gridDim = {(img_width + NUM_THREADS_X - 1) / NUM_THREADS_X, (img_height + NUM_THREADS_Y - 1) / NUM_THREADS_Y, 1};

where each individual block within the grid contains a layout of threads defined by ``const dim3 blockDim = {NUM_THREADS_X, NUM_THREADS_Y, 1};``.
An example of a kernel function that requires two dimensional grids is the main :func:`gsplat.rasterize_gaussians` kernel that renders an arbitrary number of pixels in an output image.


Config Constants
----------------

We fix the total number of threads within a block to 256. This means that for 1D blocks the block dimension is given by ``const int blockDim = 256;`` and for 2D blocks ``const dim3 blockDim = {16, 16, 1};``.
