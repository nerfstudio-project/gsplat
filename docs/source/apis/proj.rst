ProjectGaussians
===================================

.. currentmodule:: diff_rast

Given 3D gaussians parametrized by means `μ`, covariances `Σ`, colors `c`, and opacities `o`, the 
ProjectGaussians function computes the projected 2D gaussians in the camera frame with means `μ'`, covariances `Σ'`, and depths `z`
as well as their maximum radii in screen space and conic parameters. 

Note, covariances are reparametrized by the eigen decomposition 

.. math::
   
   Σ = RSS^{T}R^{T}

Where rotation matrices `R` are obtained from four dimensional quaternions.

.. autoclass:: ProjectGaussians
    :members: