ProjectGaussians
===================================

.. currentmodule:: gsplat

A 3D Gaussian is parametrized by a mean :math:`μ \in R^3`, covariance :math:`Σ \in R^{3 \times 3}`, color :math:`c \in R^3` encoded by spherical harmonics, and a opacity :math:`o \in R`. 

The :func:`gsplat.project_gaussians` function projects a 3D Gaussian into a 2D one based on the current camera view. The projected 2D Gaussian has a new mean :math:`μ' \in R^2`, covariance :math:`Σ' \in R^{2 \times 2}`, and depth :math:`z \in R` in camera coordinates. 

Note, covariances are reparametrized by their eigen decomposition:

.. math::
   
   Σ = RSS^{T}R^{T}

Where rotation matrices :math:`R \in SO(3)` are obtained from four dimensional quaternions :math:`q \in R^4`.

The projection of 3D Gaussians is approximated with the Jacobian of the perspective projection equation 
as shown in :cite:p:`zwicker2002ewa`:

.. math::

    J = \begin{bmatrix}
            f_{x}/t_{z} & 0 & -f_{x} t_{x}/t_{z}^{2} \\
            0 & f_{y}/t_{z} & -f_{y} t_{y}/t_{z}^{2} \\
            0 & 0 & 0
        \end{bmatrix}

Where :math:`t` is the center of a Gaussian in camera frame :math:`t = Wμ+p`. The projected 2D covarience is then given by: 

.. math::

    Σ' = J W Σ W^{⊤} J^{⊤}


Citations
-------------
.. bibliography::
    :style: unsrt
    :filter: docname in docnames

.. autofunction:: project_gaussians