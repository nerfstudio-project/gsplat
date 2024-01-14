RasterizeGaussians
===================================

.. currentmodule:: gsplat

Given 2D gaussians that are parametrized by their means :math:`μ'` and covariances :math:`Σ'` as well as their radii and conic parameters,
the :func:`gsplat.rasterize_gaussians` function first sorts each gaussian such that all gaussians within the bounds of a tile are grouped and sorted by increasing depth :math:`z`,
and then renders each pixel within a tile with alpha-compositing. 

The discrete rendering equation is given by: 

.. math::

    \sum_{t=n}^{N}c_{n}·α_{n}·T_{n}

Where 

.. math::

    T_{n} = \prod_{t=m}^{M}(1-α_{m})

And 

.. math::

    α_{n} = o_{n} \exp(-σ_{n})
    
    σ_{n} = \frac{1}{2} ∆^{⊤}_{n} Σ'^{−1} ∆_{n}


:math:`σ ∈ R^{2}` is the Mahalanobis distance (here referred to as sigma) which measures how many standard deviations away the center of a gaussian and the rendered pixel center is which is denoted by delta :math:`∆.`

The python bindings support conventional 3-channel RGB rasterization as well as N-dimensional rasterization with :func:`gsplat.rasterize_gaussians`.


.. autofunction:: rasterize_gaussians