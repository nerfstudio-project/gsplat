# Author: True Price <jtprice at cs.unc.edu>

import numpy as np

#-------------------------------------------------------------------------------
#
# Axis-Angle Functions
#
#-------------------------------------------------------------------------------

# returns the cross product matrix representation of a 3-vector v
def cross_prod_matrix(v):
    return np.array(((0., -v[2], v[1]), (v[2], 0., -v[0]), (-v[1], v[0], 0.)))

#-------------------------------------------------------------------------------

# www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
# if angle is None, assume ||axis|| == angle, in radians
# if angle is not None, assume that axis is a unit vector
def axis_angle_to_rotation_matrix(axis, angle=None):
    if angle is None:
        angle = np.linalg.norm(axis)
        if np.abs(angle) > np.finfo('float').eps:
            axis = axis / angle

    cp_axis = cross_prod_matrix(axis)
    return np.eye(3) + (
        np.sin(angle) * cp_axis + (1. - np.cos(angle)) * cp_axis.dot(cp_axis))

#-------------------------------------------------------------------------------

# after some deliberation, I've decided the easiest way to do this is to use
# quaternions as an intermediary
def rotation_matrix_to_axis_angle(R):
    return Quaternion.FromR(R).ToAxisAngle()

#-------------------------------------------------------------------------------
#
# Quaternion
#
#-------------------------------------------------------------------------------

class Quaternion:
    # create a quaternion from an existing rotation matrix
    # euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    @staticmethod
    def FromR(R):
        trace = np.trace(R)
  
        if trace > 0:
            qw = 0.5 * np.sqrt(1. + trace)
            qx = (R[2,1] - R[1,2]) * 0.25 / qw
            qy = (R[0,2] - R[2,0]) * 0.25 / qw
            qz = (R[1,0] - R[0,1]) * 0.25 / qw
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2. * np.sqrt(1. + R[0,0] - R[1,1] - R[2,2])
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2. * np.sqrt(1. + R[1,1] - R[0,0] - R[2,2])
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = 2. * np.sqrt(1. + R[2,2] - R[0,0] - R[1,1])
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s
  
        return Quaternion(np.array((qw, qx, qy, qz)))
  
    # if angle is None, assume ||axis|| == angle, in radians
    # if angle is not None, assume that axis is a unit vector
    @staticmethod
    def FromAxisAngle(axis, angle=None):
        if angle is None:
            angle = np.linalg.norm(axis)
            if np.abs(angle) > np.finfo('float').eps:
                axis = axis / angle

        qw = np.cos(0.5 * angle)
        axis = axis * np.sin(0.5 * angle)

        return Quaternion(np.array((qw, axis[0], axis[1], axis[2])))

    #---------------------------------------------------------------------------
  
    def __init__(self, q=np.array((1., 0., 0., 0.))):
        if isinstance(q, Quaternion):
            self.q = q.q.copy()
        else:
            q = np.asarray(q)
            if q.size == 4:
                self.q = q.copy()
            elif q.size == 3: # convert from a 3-vector to a quaternion
                self.q = np.empty(4)
                self.q[0], self.q[1:] = 0., q.ravel()
            else:
                raise Exception('Input quaternion should be a 3- or 4-vector')
  
    def __add__(self, other):
        return Quaternion(self.q + other.q)
  
    def __iadd__(self, other):
        self.q += other.q
        return self
  
    # conjugation via the ~ operator
    def __invert__(self):
        return Quaternion(
            np.array((self.q[0], -self.q[1], -self.q[2], -self.q[3])))
  
    # returns: self.q * other.q if other is a Quaternion; otherwise performs
    #          scalar multiplication
    def __mul__(self, other):
      if isinstance(other, Quaternion): # quaternion multiplication
          return Quaternion(np.array((
              self.q[0] * other.q[0] - self.q[1] * other.q[1] -
                 self.q[2] * other.q[2] - self.q[3] * other.q[3],
              self.q[0] * other.q[1] + self.q[1] * other.q[0] +
                 self.q[2] * other.q[3] - self.q[3] * other.q[2],
              self.q[0] * other.q[2] - self.q[1] * other.q[3] +
                 self.q[2] * other.q[0] + self.q[3] * other.q[1],
              self.q[0] * other.q[3] + self.q[1] * other.q[2] -
                 self.q[2] * other.q[1] + self.q[3] * other.q[0])))
      else: # scalar multiplication (assumed)
          return Quaternion(other * self.q)
  
    def __rmul__(self, other):
        return self * other
  
    def __imul__(self, other):
        self.q[:] = (self * other).q
        return self
  
    def __irmul__(self, other):
        self.q[:] = (self * other).q
        return self
  
    def __neg__(self):
        return Quaternion(-self.q)
  
    def __sub__(self, other):
        return Quaternion(self.q - other.q)
  
    def __isub__(self, other):
        self.q -= other.q
        return self
  
    def __str__(self):
        return str(self.q)

    def copy(self):
        return Quaternion(self)
  
    def dot(self, other):
        return self.q.dot(other.q)
  
    # assume the quaternion is nonzero!
    def inverse(self):
        return Quaternion((~self).q / self.q.dot(self.q))
  
    def norm(self):
        return np.linalg.norm(self.q)
  
    def normalize(self):
        self.q /= np.linalg.norm(self.q)
        return self
  
    # assume x is a Nx3 numpy array or a numpy 3-vector
    def rotate_points(self, x):
        x = np.atleast_2d(x)
        return x.dot(self.ToR().T)
  
    # convert to a rotation matrix
    def ToR(self):
        return np.eye(3) + 2 * np.array((
          (-self.q[2] * self.q[2] - self.q[3] * self.q[3],
            self.q[1] * self.q[2] - self.q[3] * self.q[0],
            self.q[1] * self.q[3] + self.q[2] * self.q[0]),
          ( self.q[1] * self.q[2] + self.q[3] * self.q[0],
           -self.q[1] * self.q[1] - self.q[3] * self.q[3],
            self.q[2] * self.q[3] - self.q[1] * self.q[0]),
          ( self.q[1] * self.q[3] - self.q[2] * self.q[0],
            self.q[2] * self.q[3] + self.q[1] * self.q[0],
           -self.q[1] * self.q[1] - self.q[2] * self.q[2])))
  
    # convert to axis-angle representation, with angle encoded by the length
    def ToAxisAngle(self):
        # recall that for axis-angle representation (a, angle), with "a" unit:
        #   q = (cos(angle/2), a * sin(angle/2))
        # below, for readability, "theta" actually means half of the angle

        sin_sq_theta = self.q[1:].dot(self.q[1:])
    
        # if theta is non-zero, then we can compute a unique rotation
        if np.abs(sin_sq_theta) > np.finfo('float').eps:
            sin_theta = np.sqrt(sin_sq_theta)
            cos_theta = self.q[0]
    
            # atan2 is more stable, so we use it to compute theta
            # note that we multiply by 2 to get the actual angle
            angle = 2. * (
                np.arctan2(-sin_theta, -cos_theta) if cos_theta < 0. else
                np.arctan2(sin_theta, cos_theta))
    
            return self.q[1:] * (angle / sin_theta)

        # otherwise, the result is singular, and we avoid dividing by
        # sin(angle/2) = 0
        return np.zeros(3)

    # euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler
    # this assumes the quaternion is non-zero
    # returns yaw, pitch, roll, with application in that order
    def ToEulerAngles(self):
        qsq = self.q**2
        k = 2. * (self.q[0] * self.q[3] + self.q[1] * self.q[2]) / qsq.sum()

        if (1. - k) < np.finfo('float').eps: # north pole singularity
            return  2. * np.arctan2(self.q[1], self.q[0]), 0.5 * np.pi, 0.
        if (1. + k) < np.finfo('float').eps: # south pole singularity
            return -2. * np.arctan2(self.q[1], self.q[0]), -0.5 * np.pi, 0.

        yaw = np.arctan2(2. * (self.q[0] * self.q[2] - self.q[1] * self.q[3]),
                         qsq[0] + qsq[1] - qsq[2] - qsq[3])
        pitch = np.arcsin(k)
        roll = np.arctan2(2. * (self.q[0] * self.q[1] - self.q[2] * self.q[3]),
                          qsq[0] - qsq[1] + qsq[2] - qsq[3])

        return yaw, pitch, roll
  
#-------------------------------------------------------------------------------
#
# DualQuaternion
#
#-------------------------------------------------------------------------------

class DualQuaternion:
    # DualQuaternion from an existing rotation + translation
    @staticmethod
    def FromQT(q, t):
        return DualQuaternion(qe=(0.5 * np.asarray(t))) * DualQuaternion(q)
  
    def __init__(self, q0=np.array((1., 0., 0., 0.)), qe=np.zeros(4)):
        self.q0, self.qe = Quaternion(q0), Quaternion(qe)
  
    def __add__(self, other):
        return DualQuaternion(self.q0 + other.q0, self.qe + other.qe)
  
    def __iadd__(self, other):
        self.q0 += other.q0
        self.qe += other.qe
        return self
  
    # conguation via the ~ operator
    def __invert__(self):
        return DualQuaternion(~self.q0, ~self.qe)
  
    def __mul__(self, other):
        if isinstance(other, DualQuaternion):
            return DualQuaternion(
                self.q0 * other.q0,
                self.q0 * other.qe + self.qe * other.q0)
        elif isinstance(other, complex): # multiplication by a dual number
            return DualQuaternion(
                self.q0 * other.real,
                self.q0 * other.imag + self.qe * other.real)
        else: # scalar multiplication (assumed)
            return DualQuaternion(other * self.q0, other * self.qe)
  
    def __rmul__(self, other):
        return self.__mul__(other)
  
    def __imul__(self, other):
        tmp = self * other
        self.q0, self.qe = tmp.q0, tmp.qe
        return self
  
    def __neg__(self):
        return DualQuaternion(-self.q0, -self.qe)
  
    def __sub__(self, other):
        return DualQuaternion(self.q0 - other.q0, self.qe - other.qe)
  
    def __isub__(self, other):
        self.q0 -= other.q0
        self.qe -= other.qe
        return self
  
    # q^-1 = q* / ||q||^2
    # assume that q0 is nonzero!
    def inverse(self):
        normsq = complex(q0.dot(q0), 2. * self.q0.q.dot(self.qe.q))
        inv_len_real = 1. / normsq.real
        return ~self * complex(
            inv_len_real, -normsq.imag * inv_len_real * inv_len_real)
  
    # returns a complex representation of the real and imaginary parts of the norm
    # assume that q0 is nonzero!
    def norm(self):
        q0_norm = self.q0.norm()
        return complex(q0_norm, self.q0.dot(self.qe) / q0_norm)
  
    # assume that q0 is nonzero!
    def normalize(self):
        # current length is ||q0|| + eps * (<q0, qe> / ||q0||)
        # writing this as a + eps * b, the inverse is
        #   1/||q|| = 1/a - eps * b / a^2
        norm = self.norm()
        inv_len_real = 1. / norm.real
        self *= complex(inv_len_real, -norm.imag * inv_len_real * inv_len_real)
        return self
  
    # return the translation vector for this dual quaternion
    def getT(self):
        return 2 * (self.qe * ~self.q0).q[1:]
  
    def ToQT(self):
        return self.q0, self.getT()
