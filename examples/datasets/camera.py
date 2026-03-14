# Author: True Price <jtprice at cs.unc.edu>

import numpy as np

from scipy.optimize import root


#-------------------------------------------------------------------------------
#
# camera distortion functions for arrays of size (..., 2)
#
#-------------------------------------------------------------------------------

def simple_radial_distortion(camera, x):
    return x * (1. + camera.k1 * np.square(x).sum(axis=-1, keepdims=True))

def radial_distortion(camera, x):
    r_sq = np.square(x).sum(axis=-1, keepdims=True)
    return x * (1. + r_sq * (camera.k1 + camera.k2 * r_sq))

def opencv_distortion(camera, x):
    x_sq = np.square(x)
    xy = np.prod(x, axis=-1, keepdims=True)
    r_sq = x_sq.sum(axis=-1, keepdims=True)

    return x * (1. + r_sq * (camera.k1 + camera.k2 * r_sq)) + np.concatenate((
        2. * camera.p1 * xy + camera.p2 * (r_sq + 2. * x_sq),
        camera.p1 * (r_sq + 2. * y_sq) + 2. * camera.p2 * xy),
        axis=-1)


#-------------------------------------------------------------------------------
#
# Camera
#
#-------------------------------------------------------------------------------

class Camera:
    @staticmethod
    def GetNumParams(type_):
        if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
            return 3
        if type_ == 1 or type_ == 'PINHOLE':
            return 4
        if type_ == 2 or type_ == 'SIMPLE_RADIAL':
            return 4
        if type_ == 3 or type_ == 'RADIAL':
            return 5
        if type_ == 4 or type_ == 'OPENCV':
            return 8
        if type_ == 5 or type_ == 'OPENCV_FISHEYE':
           return 8
        #if type_ == 6 or type_ == 'FULL_OPENCV':
        #    return 12
        #if type_ == 7 or type_ == 'FOV':
        #    return 5
        #if type_ == 8 or type_ == 'SIMPLE_RADIAL_FISHEYE':
        #    return 4
        #if type_ == 9 or type_ == 'RADIAL_FISHEYE':
        #    return 5
        #if type_ == 10 or type_ == 'THIN_PRISM_FISHEYE':
        #    return 12

        # TODO: not supporting other camera types, currently
        raise Exception('Camera type not supported')


    #---------------------------------------------------------------------------

    @staticmethod
    def GetNameFromType(type_):
        if type_ == 0: return 'SIMPLE_PINHOLE'
        if type_ == 1: return 'PINHOLE'
        if type_ == 2: return 'SIMPLE_RADIAL'
        if type_ == 3: return 'RADIAL'
        if type_ == 4: return 'OPENCV'
        if type_ == 5: return 'OPENCV_FISHEYE'
        #if type_ == 6: return 'FULL_OPENCV'
        #if type_ == 7: return 'FOV'
        #if type_ == 8: return 'SIMPLE_RADIAL_FISHEYE'
        #if type_ == 9: return 'RADIAL_FISHEYE'
        #if type_ == 10: return 'THIN_PRISM_FISHEYE'

        raise Exception('Camera type not supported')


    #---------------------------------------------------------------------------

    def __init__(self, type_, width_, height_, params):
        self.width = width_
        self.height = height_

        if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
            self.fx, self.cx, self.cy = params
            self.fy = self.fx
            self.distortion_func = None
            self.camera_type = 0

        elif type_ == 1 or type_ == 'PINHOLE':
            self.fx, self.fy, self.cx, self.cy = params
            self.distortion_func = None
            self.camera_type = 1

        elif type_ == 2 or type_ == 'SIMPLE_RADIAL':
            self.fx, self.cx, self.cy, self.k1 = params
            self.fy = self.fx
            self.distortion_func = simple_radial_distortion
            self.camera_type = 2

        elif type_ == 3 or type_ == 'RADIAL':
            self.fx, self.cx, self.cy, self.k1, self.k2 = params
            self.fy = self.fx
            self.distortion_func = radial_distortion
            self.camera_type = 3

        elif type_ == 4 or type_ == 'OPENCV':
            self.fx, self.fy, self.cx, self.cy = params[:4]
            self.k1, self.k2, self.p1, self.p2 = params[4:]
            self.distortion_func = opencv_distortion
            self.camera_type = 4

        elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
            self.fx, self.fy, self.cx, self.cy = params[:4]
            self.k1, self.k2, self.k3, self.k4 = params[4:]
            def fn(camera, x):
                raise Exception('Fisheye distortion not supported')
            self.distortion_func = fn
            self.camera_type = 5

        else:
            raise Exception('Camera type not supported')


    #---------------------------------------------------------------------------

    def __str__(self):
        s = (self.GetNameFromType(self.camera_type) +
             ' {} {} {}'.format(self.width, self.height, self.fx))

        if self.camera_type in (1, 4): # PINHOLE, OPENCV
            s += ' {}'.format(self.fy)

        s += ' {} {}'.format(self.cx, self.cy)

        if self.camera_type == 2: # SIMPLE_RADIAL
            s += ' {}'.format(self.k1)

        elif self.camera_type == 3: # RADIAL
            s += ' {} {}'.format(self.k1, self.k2)

        elif self.camera_type == 4: # OPENCV
            s += ' {} {} {} {}'.format(self.k1, self.k2, self.p1, self.p2)

        elif self.camera_type == 5:  # OPENCV_FISHEYE
            s += ' {} {} {} {}'.format(self.k1, self.k2, self.k3, self.k4)

        return s


    #---------------------------------------------------------------------------

    # return the camera parameters in the same order as the colmap output format
    def get_params(self):
        if self.camera_type == 0:
            return np.array((self.fx, self.cx, self.cy))
        if self.camera_type == 1:
            return np.array((self.fx, self.fy, self.cx, self.cy))
        if self.camera_type == 2:
            return np.array((self.fx, self.cx, self.cy, self.k1))
        if self.camera_type == 3:
            return np.array((self.fx, self.cx, self.cy, self.k1, self.k2))
        if self.camera_type == 4:
            return np.array((self.fx, self.fy, self.cx, self.cy, self.k1,
                             self.k2, self.p1, self.p2))
        if self.camera_type == 5:
            return np.array((self.fx, self.fy, self.cx, self.cy, self.k1, self.k2,
                            self.k3, self.k4))


    #---------------------------------------------------------------------------

    def get_camera_matrix(self):
        return np.array(
            ((self.fx, 0, self.cx), (0, self.fy, self.cy), (0, 0, 1)))

    def get_inverse_camera_matrix(self):
        return np.array(
            ((1. / self.fx, 0, -self.cx / self.fx),
             (0, 1. / self.fy, -self.cy / self.fy),
             (0, 0, 1)))

    @property
    def K(self):
        return self.get_camera_matrix()

    @property
    def K_inv(self):
        return self.get_inverse_camera_matrix()

    #---------------------------------------------------------------------------

    # return the inverse camera matrix
    def get_inv_camera_matrix(self):
        inv_fx, inv_fy = 1. / self.fx, 1. / self.fy
        return np.array(((inv_fx, 0, -inv_fx * self.cx),
                         (0, inv_fy, -inv_fy * self.cy),
                         (0, 0, 1)))


    #---------------------------------------------------------------------------

    # return an (x, y) pixel coordinate grid for this camera
    def get_image_grid(self):
        xmin = (0.5 - self.cx) / self.fx
        xmax = (self.width - 0.5 - self.cx) / self.fx
        ymin = (0.5 - self.cy) / self.fy
        ymax = (self.height - 0.5 - self.cy) / self.fy
        return np.meshgrid(np.linspace(xmin, xmax, self.width),
                           np.linspace(ymin, ymax, self.height))


    #---------------------------------------------------------------------------

    # x: array of shape (N,2) or (2,)
    # normalized: False if the input points are in pixel coordinates
    # denormalize: True if the points should be put back into pixel coordinates
    def distort_points(self, x, normalized=True, denormalize=True):
        x = np.atleast_2d(x)

        # put the points into normalized camera coordinates
        if not normalized:
            x -= np.array([[self.cx, self.cy]])
            x /= np.array([[self.fx, self.fy]])

        # distort, if necessary
        if self.distortion_func is not None:
            x = self.distortion_func(self, x)

        if denormalize:
            x *= np.array([[self.fx, self.fy]])
            x += np.array([[self.cx, self.cy]])

        return x


    #---------------------------------------------------------------------------

    # x: array of shape (N1,N2,...,2), (N,2), or (2,)
    # normalized: False if the input points are in pixel coordinates
    # denormalize: True if the points should be put back into pixel coordinates
    def undistort_points(self, x, normalized=False, denormalize=True):
        x = np.atleast_2d(x)

        # put the points into normalized camera coordinates
        if not normalized:
            x = x - np.array([self.cx, self.cy]) # creates a copy
            x /= np.array([self.fx, self.fy])

        # undistort, if necessary
        if self.distortion_func is not None:
            def objective(xu):
                return (x - self.distortion_func(self, xu.reshape(*x.shape))
                    ).ravel()

            xu = root(objective, x).x.reshape(*x.shape)
        else:
            xu = x
            
        if denormalize:
            xu *= np.array([[self.fx, self.fy]])
            xu += np.array([[self.cx, self.cy]])

        return xu
