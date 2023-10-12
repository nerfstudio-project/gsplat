Data Conventions
===================================

.. currentmodule:: gsplat

Here we explain the various data conventions used in our repo.

Rotation Convention
-------------------
We represent rotations with four dimensional vectors :math:`q = (w,x,y,z)` such that the 3x3 :math:`SO(3)` rotation matrix is defined by:

.. math::

    R = \begin{bmatrix}
        1 - 2 \left( y^2 + z^2 \right) & 2 \left( x y - w z \right) & 2 \left( x z + w y \right) \\
        2 \left( x y + w z \right) & 1 - 2 \left( x^2 + z^2 \right) & 2 \left( y z - w x \right) \\
        2 \left( x z - w y \right) & 2 \left( y z + w x \right) & 1 - 2 \left( x^2 + y^2 \right) \\
        \end{bmatrix}

View Matrix and Projection Matrix
---------------------------------
We refer to the `view matrix` :math:`W` as the world to camera frame transformation (referred to as `w2c` in some sources) that maps
3D world points :math:`(x,y,z)_{world}` to 3D camera points :math:`(x,y,z)_{cam}` where :math:`z_{cam}` is the relative depth to the camera center.


The `projection matrix` refers to the full projective transformation that maps 3D points in the world frame to the 2D points in the image/pixel frame.
This transformation is the concatenation of the perspective projection matrix :math:`K` (obtained from camera intrinsics) and the view matrix :math:`W`.
We adopt the `OpenGL <http://www.songho.ca/opengl/gl_projectionmatrix.html>`_ perspective projection convention. The projection matrix :math:`P` is given by:

.. math:: 

    P = K W

.. math::

    K = \begin{bmatrix}
        \frac{2n}{r - l} & 0.0 & \frac{r + l}{r - l} & 0.0 \\
        0.0 & \frac{2n}{t - b} & \frac{t + b}{t - b} & 0.0 \\
        0.0 & 0.0 & \frac{f + n}{f - n} & -\frac{f \cdot n}{f - n} \\
        0.0 & 0.0 & 1.0 & 0.0 \\
    \end{bmatrix}
