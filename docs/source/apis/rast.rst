RasterizeGaussians
===================================

.. currentmodule:: gsplat

Given 2D Gaussians that are parametrized by their means :math:`μ'`, covariances :math:`Σ'`, and depths :math:`z`,
the :func:`gsplat.rasterize_gaussians` function gives a unique ID to each Gaussian depending on what tile it hits and its depth value, sorts all Gaussians by depth in a single global sort,
and then renders each pixel within a tile by alpha-compositing each Gaussian ovelapping the pixel. 

The color at a pixel :math:`\hat{C}` is given by the discrete volume rendering equation: 

.. math::

    \hat{C}  = \sum_{i \in N} c_{i} \alpha_{i} T_{i}

Where 

.. math::

    T_{i} = \prod_{j=1}^{i-1}(1-\alpha_{j})

is the accumulated transmittance at the pixel :math:`p` and 

.. math::

    {\alpha_i}  = o_i \cdot \exp{\left(\frac{1}{2}({p}-{\mu}_i)^\intercal {\Sigma}_i^{-1}({p}-{\mu}_i)\right)}


is referred to as the alpha or density term in some work. 

The Python bindings support conventional 3-channel RGB rasterization as well as N-dimensional rasterization with :func:`gsplat.rasterize_gaussians`.


.. autofunction:: rasterize_gaussians