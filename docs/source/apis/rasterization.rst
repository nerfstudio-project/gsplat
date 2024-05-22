Rasterization
===================================

.. currentmodule:: gsplat

Given a set of 3D gaussians parametrized by means :math:`\mu \in \mathbb{R}^3`, covariances 
:math:`\Sigma \in \mathbb{R}^{3 \times 3}`, colors :math:`c`, and opacities :math:`o`, we first 
compute their projected means :math:`\mu' \in \mathbb{R}^2` and covariances 
:math:`\Sigma' \in \mathbb{R}^{2 \times 2}` on the image planes. Then we sort each gaussian such 
that all gaussians within the bounds of a tile are grouped and sorted by increasing 
depth :math:`z`, and then render each pixel within the tile with alpha-compositing. 

Note, the 3D covariances are reparametrized with a scaling matrix 
:math:`S = \text{diag}(\mathbf{s}) \in \mathbb{R}^{3 \times 3}` represented by a 
scale vector :math:`s \in \mathbb{R}^3`, and a rotation matrix 
:math:`R \in \mathbb{R}^{3 \times 3}` represented by a rotation 
quaternion :math:`q \in \mathcal{R}^4`:

.. math::
   
   \Sigma = RSS^{T}R^{T}

The projection of 3D Gaussians is approximated with the Jacobian of the perspective 
projection equation:

.. math::

    J = \begin{bmatrix}
        f_{x}/z & 0 & -f_{x} t_{x}/z^{2} \\
        0 & f_{y}/z & -f_{y} t_{y}/z^{2} \\
        0 & 0 & 0
    \end{bmatrix}

.. math::

    \Sigma' = J W \Sigma W^{T} J^{T}

Where :math:`[W | t]` is the world-to-camera transformation matrix, and :math:`f_{x}, f_{y}`
are the focal lengths of the camera.

.. The discrete rendering equation is given by: 

.. .. math::

..     \sum_{t=n}^{N}c_{n} \alpha_{n} T_{n}

.. Where 

.. .. math::

..     T_{n} = \prod_{t=m}^{M}(1-\alpha_{m})

.. And 

.. .. math::

..     \alpha = o \exp(-\sigma)
    
..     \sigma = \frac{1}{2} \Delta^{T} \Sigma'^{-1} \Delta


.. Where :math:`\sigma \in \mathbb{R}^2` is the Mahalanobis distance which measures how many 
.. standard deviations away the center of a gaussian and the rendered pixel center 
.. is which is denoted by delta :math:`\Delta`.


.. autofunction:: rasterization