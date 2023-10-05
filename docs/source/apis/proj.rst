ProjectGaussians
===================================

.. currentmodule:: diff_rast

Given 3D gaussians parametrized by means :math:`μ`, covariances :math:`Σ`, colors :math:`c`, and opacities :math:`o`, the 
ProjectGaussians function computes the projected 2D gaussians in the camera frame with means :math:`μ'`, covariances :math:`Σ'`, and depths :math:`z`
as well as their maximum radii in screen space and conic parameters. 

Note, covariances are reparametrized by the eigen decomposition:

.. math::
   
   Σ = RSS^{T}R^{T}

Where rotation matrices :math:`R` are obtained from four dimensional quaternions.

The projection of 3D Gaussians is approximated with the Jacobian of the perspective projection equation 
as shown in :cite:p:`zwicker2002ewa`:

.. math::

    J = \begin{bmatrix}
            f_{x}/t_{z} & 0 & -f_{x} t_{x}/t_{z}^{2} \\
            0 & f_{y}/t_{z} & -f_{y} t_{y}/t_{z}^{2} \\
            0 & 0 & 0
        \end{bmatrix}

Where :math:`t` is the center of a gaussian in camera frame :math:`t = Wμ+p`. The projected 2D covarience is then given by: 

.. math::

    Σ' = J W Σ W^{⊤} J^{⊤}


Citations
-------------
.. bibliography::
    :style: unsrt
    :filter: docname in docnames

.. autoclass:: ProjectGaussians