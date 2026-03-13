# Author: True Price <jtprice at cs.unc.edu>

import array
import numpy as np
import os
import struct

from collections import OrderedDict
from itertools import combinations

from .camera import Camera
from .image import Image
from .rotation import Quaternion

#-------------------------------------------------------------------------------
#
# SceneManager
#
#-------------------------------------------------------------------------------

class SceneManager:
    INVALID_POINT3D = np.uint64(-1)

    def __init__(self, colmap_results_folder, image_path=None):
        self.folder = colmap_results_folder
        if not self.folder.endswith('/'):
            self.folder += '/'

        self.image_path = None
        self.load_colmap_project_file(image_path=image_path)

        self.cameras = OrderedDict()
        self.images = OrderedDict()
        self.name_to_image_id = dict()

        self.last_camera_id = 0
        self.last_image_id = 0

        # Nx3 array of point3D xyz's
        self.points3D = np.zeros((0, 3))

        # for each element in points3D, stores the id of the point
        self.point3D_ids = np.empty(0)

        # point3D_id => index in self.points3D
        self.point3D_id_to_point3D_idx = dict()

        # point3D_id => [(image_id, point2D idx in image)]
        self.point3D_id_to_images = dict()

        self.point3D_colors = np.zeros((0, 3), dtype=np.uint8)
        self.point3D_errors = np.zeros(0)

    #---------------------------------------------------------------------------

    def load_colmap_project_file(self, project_file=None, image_path=None):
        if project_file is None:
            project_file = self.folder + 'project.ini'

        self.image_path = image_path

        if self.image_path is None:
            try:
                with open(project_file, 'r') as f:
                    for line in iter(f.readline, ''):
                        if line.startswith('image_path'):
                            self.image_path = line[11:].strip()
                            break
            except:
                pass

        if self.image_path is None:
            print('Warning: image_path not found for reconstruction')
        elif not self.image_path.endswith('/'):
            self.image_path += '/'

    #---------------------------------------------------------------------------

    def load(self):
        self.load_cameras()
        self.load_images()
        self.load_points3D()

    #---------------------------------------------------------------------------

    def load_cameras(self, input_file=None):
        if input_file is None:
            input_file = self.folder + 'cameras.bin'
            if os.path.exists(input_file):
                self._load_cameras_bin(input_file)
            else:
                input_file = self.folder + 'cameras.txt'
                if os.path.exists(input_file):
                    self._load_cameras_txt(input_file)
                else:
                    raise IOError('no cameras file found')
    
    def _load_cameras_bin(self, input_file):
        self.cameras = OrderedDict()

        with open(input_file, 'rb') as f:
            num_cameras = struct.unpack('L', f.read(8))[0]

            for _ in range(num_cameras):
                camera_id, camera_type, w, h = struct.unpack('IiLL', f.read(24))
                num_params = Camera.GetNumParams(camera_type)
                params = struct.unpack('d' * num_params, f.read(8 * num_params))
                self.cameras[camera_id] = Camera(camera_type, w, h, params)
                self.last_camera_id = max(self.last_camera_id, camera_id)

    def _load_cameras_txt(self, input_file):
        self.cameras = OrderedDict()

        with open(input_file, 'r') as f:
            for line in iter(lambda: f.readline().strip(), ''):
                if not line or line.startswith('#'):
                    continue

                data = line.split()
                camera_id = int(data[0])
                self.cameras[camera_id] = Camera(
                    data[1], int(data[2]), int(data[3]), map(float, data[4:]))
                self.last_camera_id = max(self.last_camera_id, camera_id)

    #---------------------------------------------------------------------------

    def load_images(self, input_file=None):
        if input_file is None:
            input_file = self.folder + 'images.bin'
            if os.path.exists(input_file):
                self._load_images_bin(input_file)
            else:
                input_file = self.folder + 'images.txt'
                if os.path.exists(input_file):
                    self._load_images_txt(input_file)
                else:
                    raise IOError('no images file found')

    def _load_images_bin(self, input_file):
        self.images = OrderedDict()

        with open(input_file, 'rb') as f:
            num_images = struct.unpack('L', f.read(8))[0]
            image_struct = struct.Struct('<I 4d 3d I')
            for _ in range(num_images):
                data = image_struct.unpack(f.read(image_struct.size))
                image_id = data[0]
                q = Quaternion(np.array(data[1:5]))
                t = np.array(data[5:8])
                camera_id = data[8]
                name = b''.join(c for c in iter(lambda: f.read(1), b'\x00')).decode()

                image = Image(name, camera_id, q, t)
                num_points2D = struct.unpack('Q', f.read(8))[0]

                # Optimized code below.
                # Read all elements as double first, then convert to array, slice it
                # into points2d and ids, and convert ids back to unsigned long longs
                # ('Q'). This is significantly faster than using O(num_points2D) f.read
                # calls, experiments show >7x improvements in 60 image model, 23s -> 3s.
                points_array = array.array('d')
                points_array.fromfile(f, 3 * num_points2D)
                points_elements = np.array(points_array).reshape((num_points2D, 3))
                image.points2D = points_elements[:, :2]

                ids_array = array.array('Q')
                ids_array.frombytes(points_elements[:, 2].tobytes())
                image.point3D_ids = np.array(ids_array, dtype=np.uint64).reshape(
                    (num_points2D,))

                # automatically remove points without an associated 3D point
                #mask = (image.point3D_ids != SceneManager.INVALID_POINT3D)
                #image.points2D = image.points2D[mask]
                #image.point3D_ids = image.point3D_ids[mask]

                self.images[image_id] = image
                self.name_to_image_id[image.name] = image_id

                self.last_image_id = max(self.last_image_id, image_id)

    def _load_images_txt(self, input_file):
        self.images = OrderedDict()

        with open(input_file, 'r') as f:
            is_camera_description_line = False

            for line in iter(lambda: f.readline().strip(), ''):
                if not line or line.startswith('#'):
                    continue

                is_camera_description_line = not is_camera_description_line

                data = line.split()

                if is_camera_description_line:
                    image_id = int(data[0])
                    image = Image(data[-1], int(data[-2]),
                                  Quaternion(np.array(map(float, data[1:5]))),
                                  np.array(map(float, data[5:8])))
                else:
                    image.points2D = np.array(
                        [map(float, data[::3]), map(float, data[1::3])]).T
                    image.point3D_ids = np.array(map(np.uint64, data[2::3]))

                    # automatically remove points without an associated 3D point
                    #mask = (image.point3D_ids != SceneManager.INVALID_POINT3D)
                    #image.points2D = image.points2D[mask]
                    #image.point3D_ids = image.point3D_ids[mask]

                    self.images[image_id] = image
                    self.name_to_image_id[image.name] = image_id

                    self.last_image_id = max(self.last_image_id, image_id)

    #---------------------------------------------------------------------------

    def load_points3D(self, input_file=None):
        if input_file is None:
            input_file = self.folder + 'points3D.bin'
            if os.path.exists(input_file):
                self._load_points3D_bin(input_file)
            else:
                input_file = self.folder + 'points3D.txt'
                if os.path.exists(input_file):
                    self._load_points3D_txt(input_file)
                else:
                    raise IOError('no points3D file found')

    def _load_points3D_bin(self, input_file):
        with open(input_file, 'rb') as f:
            num_points3D = struct.unpack('L', f.read(8))[0]

            self.points3D = np.empty((num_points3D, 3))
            self.point3D_ids = np.empty(num_points3D, dtype=np.uint64)
            self.point3D_colors = np.empty((num_points3D, 3), dtype=np.uint8)
            self.point3D_id_to_point3D_idx = dict()
            self.point3D_id_to_images = dict()
            self.point3D_errors = np.empty(num_points3D)

            data_struct = struct.Struct('<Q 3d 3B d Q')

            for i in range(num_points3D):
                data = data_struct.unpack(f.read(data_struct.size))
                self.point3D_ids[i] = data[0]
                self.points3D[i] = data[1:4]
                self.point3D_colors[i] = data[4:7]
                self.point3D_errors[i] = data[7]
                track_len = data[8]

                self.point3D_id_to_point3D_idx[self.point3D_ids[i]] = i

                data = struct.unpack(f'{2*track_len}I', f.read(2 * track_len * 4))

                self.point3D_id_to_images[self.point3D_ids[i]] = \
                    np.array(data, dtype=np.uint32).reshape(track_len, 2)

    def _load_points3D_txt(self, input_file):
        self.points3D = []
        self.point3D_ids = []
        self.point3D_colors = []
        self.point3D_id_to_point3D_idx = dict()
        self.point3D_id_to_images = dict()
        self.point3D_errors = []

        with open(input_file, 'r') as f:
            for line in iter(lambda: f.readline().strip(), ''):
                if not line or line.startswith('#'):
                    continue

                data = line.split()
                point3D_id = np.uint64(data[0])

                self.point3D_ids.append(point3D_id)
                self.point3D_id_to_point3D_idx[point3D_id] = len(self.points3D)
                self.points3D.append(map(np.float64, data[1:4]))
                self.point3D_colors.append(map(np.uint8, data[4:7]))
                self.point3D_errors.append(np.float64(data[7]))

                # load (image id, point2D idx) pairs
                self.point3D_id_to_images[point3D_id] = \
                    np.array(map(np.uint32, data[8:])).reshape(-1, 2)

        self.points3D = np.array(self.points3D)
        self.point3D_ids = np.array(self.point3D_ids)
        self.point3D_colors = np.array(self.point3D_colors)
        self.point3D_errors = np.array(self.point3D_errors)

    #---------------------------------------------------------------------------

    def save(self, output_folder, binary=True):
        self.save_cameras(output_folder, binary=binary)
        self.save_images(output_folder, binary=binary)
        self.save_points3D(output_folder, binary=binary)

    #---------------------------------------------------------------------------

    def save_cameras(self, output_folder, output_file=None, binary=True):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if output_file is None:
            output_file = 'cameras.bin' if binary else 'cameras.txt'

        output_file = os.path.join(output_folder, output_file)

        if binary:
            self._save_cameras_bin(output_file)
        else:
            self._save_cameras_txt(output_file)
    
    def _save_cameras_bin(self, output_file):
        with open(output_file, 'wb') as fid:
            fid.write(struct.pack('L', len(self.cameras)))

            camera_struct = struct.Struct('IiLL')

            for camera_id, camera in sorted(self.cameras.iteritems()):
                fid.write(camera_struct.pack(
                    camera_id, camera.camera_type, camera.width, camera.height))
                # TODO (True): should move this into the Camera class
                fid.write(camera.get_params().tobytes())

    def _save_cameras_txt(self, output_file):
        with open(output_file, 'w') as fid:
            print>>fid, '# Camera list with one line of data per camera:'
            print>>fid, '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]'
            print>>fid, '# Number of cameras:', len(self.cameras)

            for camera_id, camera in sorted(self.cameras.iteritems()):
                print>>fid, camera_id, camera

    #---------------------------------------------------------------------------

    def save_images(self, output_folder, output_file=None, binary=True):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if output_file is None:
            output_file = 'images.bin' if binary else 'images.txt'

        output_file = os.path.join(output_folder, output_file)

        if binary:
            self._save_images_bin(output_file)
        else:
            self._save_images_txt(output_file)

    def _save_images_bin(self, output_file):
        with open(output_file, 'wb') as fid:
            fid.write(struct.pack('L', len(self.images)))

            for image_id, image in self.images.iteritems():
                fid.write(struct.pack('I', image_id))
                fid.write(image.q.q.tobytes())
                fid.write(image.tvec.tobytes())
                fid.write(struct.pack('I', image.camera_id))
                fid.write(image.name + '\0')
                fid.write(struct.pack('L', len(image.points2D)))
                data = np.rec.fromarrays(
                    (image.points2D[:,0], image.points2D[:,1], image.point3D_ids))
                fid.write(data.tobytes())

    def _save_images_txt(self, output_file):
        with open(output_file, 'w') as fid:
            print>>fid, '# Image list with two lines of data per image:'
            print>>fid, '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME'
            print>>fid, '#   POINTS2D[] as (X, Y, POINT3D_ID)'
            print>>fid, '# Number of images: {},'.format(len(self.images)),
            print>>fid, 'mean observations per image: unknown'

            for image_id, image in self.images.iteritems():
                print>>fid, image_id,
                print>>fid, ' '.join(str(qi) for qi in image.q.q),
                print>>fid, ' '.join(str(ti) for ti in image.tvec),
                print>>fid, image.camera_id, image.name

                data = np.rec.fromarrays(
                    (image.points2D[:,0], image.points2D[:,1],
                     image.point3D_ids.astype(np.int64)))
                if len(data) > 0:
                    np.savetxt(fid, data, '%.2f %.2f %d', newline=' ')
                    fid.seek(-1, os.SEEK_CUR)
                fid.write('\n')

    #---------------------------------------------------------------------------

    def save_points3D(self, output_folder, output_file=None, binary=True):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if output_file is None:
            output_file = 'points3D.bin' if binary else 'points3D.txt'

        output_file = os.path.join(output_folder, output_file)

        if binary:
            self._save_points3D_bin(output_file)
        else:
            self._save_points3D_txt(output_file)

    def _save_points3D_bin(self, output_file):
        num_valid_points3D = sum(
            1 for point3D_idx in self.point3D_id_to_point3D_idx.itervalues()
            if point3D_idx != SceneManager.INVALID_POINT3D)

        iter_point3D_id_to_point3D_idx = \
            self.point3D_id_to_point3D_idx.iteritems()

        with open(output_file, 'wb') as fid:
            fid.write(struct.pack('L', num_valid_points3D))

            for point3D_id, point3D_idx in iter_point3D_id_to_point3D_idx:
                if point3D_idx == SceneManager.INVALID_POINT3D:
                    continue

                fid.write(struct.pack('L', point3D_id))
                fid.write(self.points3D[point3D_idx].tobytes())
                fid.write(self.point3D_colors[point3D_idx].tobytes())
                fid.write(self.point3D_errors[point3D_idx].tobytes())
                fid.write(
                    struct.pack('L', len(self.point3D_id_to_images[point3D_id])))
                fid.write(self.point3D_id_to_images[point3D_id].tobytes())

    def _save_points3D_txt(self, output_file):
        num_valid_points3D = sum(
            1 for point3D_idx in self.point3D_id_to_point3D_idx.itervalues()
            if point3D_idx != SceneManager.INVALID_POINT3D)

        array_to_string = lambda arr: ' '.join(str(x) for x in arr)

        iter_point3D_id_to_point3D_idx = \
            self.point3D_id_to_point3D_idx.iteritems()

        with open(output_file, 'w') as fid:
            print>>fid, '# 3D point list with one line of data per point:'
            print>>fid, '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as ',
            print>>fid, '(IMAGE_ID, POINT2D_IDX)'
            print>>fid, '# Number of points: {},'.format(num_valid_points3D),
            print>>fid, 'mean track length: unknown'

            for point3D_id, point3D_idx in iter_point3D_id_to_point3D_idx:
                if point3D_idx == SceneManager.INVALID_POINT3D:
                    continue

                print>>fid, point3D_id,
                print>>fid, array_to_string(self.points3D[point3D_idx]),
                print>>fid, array_to_string(self.point3D_colors[point3D_idx]),
                print>>fid, self.point3D_errors[point3D_idx],
                print>>fid, array_to_string(
                    self.point3D_id_to_images[point3D_id].flat)

    #---------------------------------------------------------------------------

    # return the image id associated with a given image file
    def get_image_from_name(self, image_name):
        image_id = self.name_to_image_id[image_name]
        return image_id, self.images[image_id]

    #---------------------------------------------------------------------------

    def get_camera(self, camera_id):
        return self.cameras[camera_id]

    #---------------------------------------------------------------------------

    def get_points3D(self, image_id, return_points2D=True, return_colors=False):
        image = self.images[image_id]

        mask = (image.point3D_ids != SceneManager.INVALID_POINT3D)

        point3D_idxs = np.array([
            self.point3D_id_to_point3D_idx[point3D_id]
            for point3D_id in image.point3D_ids[mask]])
        # detect filtered points
        filter_mask = (point3D_idxs != SceneManager.INVALID_POINT3D)
        point3D_idxs = point3D_idxs[filter_mask]
        result = [self.points3D[point3D_idxs,:]]

        if return_points2D:
            mask[mask] &= filter_mask
            result += [image.points2D[mask]]
        if return_colors:
            result += [self.point3D_colors[point3D_idxs,:]]

        return result if len(result) > 1 else result[0]

    #---------------------------------------------------------------------------

    def point3D_valid(self, point3D_id):
        return (self.point3D_id_to_point3D_idx[point3D_id] !=
                SceneManager.INVALID_POINT3D)

    #---------------------------------------------------------------------------

    def get_filtered_points3D(self, return_colors=False):
        point3D_idxs = [
            idx for idx in self.point3D_id_to_point3D_idx.values()
            if idx != SceneManager.INVALID_POINT3D]
        result = [self.points3D[point3D_idxs,:]]
        
        if return_colors:
            result += [self.point3D_colors[point3D_idxs,:]]

        return result if len(result) > 1 else result[0]

    #---------------------------------------------------------------------------

    # return 3D points shared by two images
    def get_shared_points3D(self, image_id1, image_id2):
        point3D_ids = (
                set(self.images[image_id1].point3D_ids) &
                set(self.images[image_id2].point3D_ids))
        point3D_ids.discard(SceneManager.INVALID_POINT3D)

        point3D_idxs = np.array([self.point3D_id_to_point3D_idx[point3D_id]
            for point3D_id in point3D_ids])

        return self.points3D[point3D_idxs,:]

    #---------------------------------------------------------------------------

    # project *all* 3D points into image, return their projection coordinates,
    # as well as their 3D positions
    def get_viewed_points(self, image_id):
        image = self.images[image_id]

        # get unfiltered points
        point3D_idxs = set(self.point3D_id_to_point3D_idx.itervalues())
        point3D_idxs.discard(SceneManager.INVALID_POINT3D)
        point3D_idxs = list(point3D_idxs)
        points3D = self.points3D[point3D_idxs,:]

        # orient points relative to camera
        R = image.q.ToR()
        points3D = points3D.dot(R.T) + image.tvec[np.newaxis,:]
        points3D = points3D[points3D[:,2] > 0,:] # keep points with positive z

        # put points into image coordinates
        camera = self.cameras[image.camera_id]
        points2D = points3D.dot(camera.get_camera_matrix().T)
        points2D = points2D[:,:2] / points2D[:,2][:,np.newaxis]

        # keep points that are within the image
        mask = (
            (points2D[:,0] >= 0) &
            (points2D[:,1] >= 0) &
            (points2D[:,0] < camera.width - 1) &
            (points2D[:,1] < camera.height - 1))

        return points2D[mask,:], points3D[mask,:]

    #---------------------------------------------------------------------------

    def add_camera(self, camera):
        self.last_camera_id += 1
        self.cameras[self.last_camera_id] = camera
        return self.last_camera_id

    #---------------------------------------------------------------------------

    def add_image(self, image):
        self.last_image_id += 1
        self.images[self.last_image_id] = image
        return self.last_image_id

    #---------------------------------------------------------------------------

    def delete_images(self, image_list):
        # delete specified images
        for image_id in image_list:
            if image_id in self.images:
                del self.images[image_id]

        keep_set = set(self.images.iterkeys())

        # delete references to specified images, and ignore any points that are
        # invalidated
        iter_point3D_id_to_point3D_idx = \
            self.point3D_id_to_point3D_idx.iteritems()

        for point3D_id, point3D_idx in iter_point3D_id_to_point3D_idx:
            if point3D_idx == SceneManager.INVALID_POINT3D:
                continue

            mask = np.array([
                image_id in keep_set
                for image_id in self.point3D_id_to_images[point3D_id][:,0]])
            if np.any(mask):
                self.point3D_id_to_images[point3D_id] = \
                    self.point3D_id_to_images[point3D_id][mask]
            else:
                self.point3D_id_to_point3D_idx[point3D_id] = \
                    SceneManager.INVALID_POINT3D

    #---------------------------------------------------------------------------

    # camera_list: set of cameras whose points we'd like to keep
    # min/max triangulation angle: in degrees
    def filter_points3D(self,
            min_track_len=0, max_error=np.inf, min_tri_angle=0,
            max_tri_angle=180, image_set=set()):

        image_set = set(image_set)

        check_triangulation_angles = (min_tri_angle > 0 or max_tri_angle < 180)
        if check_triangulation_angles:
            max_tri_prod = np.cos(np.radians(min_tri_angle))
            min_tri_prod = np.cos(np.radians(max_tri_angle))

        iter_point3D_id_to_point3D_idx = \
            self.point3D_id_to_point3D_idx.iteritems()

        image_ids = []

        for point3D_id, point3D_idx in iter_point3D_id_to_point3D_idx:
            if point3D_idx == SceneManager.INVALID_POINT3D:
                continue

            if image_set or min_track_len > 0:
                image_ids = set(self.point3D_id_to_images[point3D_id][:,0])
            
            # check if error and min track length are sufficient, or if none of
            # the selected cameras see the point
            if (len(image_ids) < min_track_len or
                      self.point3D_errors[point3D_idx] > max_error or
                      image_set and image_set.isdisjoint(image_ids)):
                self.point3D_id_to_point3D_idx[point3D_id] = \
                    SceneManager.INVALID_POINT3D

            # find dot product between all camera viewing rays
            elif check_triangulation_angles:
                xyz = self.points3D[point3D_idx,:]
                tvecs = np.array(
                    [(self.images[image_id].tvec - xyz)
                     for image_id in image_ids])
                tvecs /= np.linalg.norm(tvecs, axis=-1)[:,np.newaxis]

                cos_theta = np.array(
                    [u.dot(v) for u,v in combinations(tvecs, 2)])

                # min_prod = cos(maximum viewing angle), and vice versa
                # if maximum viewing angle is too small or too large,
                # don't add this point
                if (np.min(cos_theta) > max_tri_prod or
                    np.max(cos_theta) < min_tri_prod):
                    self.point3D_id_to_point3D_idx[point3D_id] = \
                        SceneManager.INVALID_POINT3D

        # apply the filters to the image point3D_ids
        for image in self.images.itervalues():
            mask = np.array([
                self.point3D_id_to_point3D_idx.get(point3D_id, 0) \
                    == SceneManager.INVALID_POINT3D
                for point3D_id in image.point3D_ids])
            image.point3D_ids[mask] = SceneManager.INVALID_POINT3D

    #---------------------------------------------------------------------------

    # scene graph: {image_id: [image_id: #shared points]}
    def build_scene_graph(self):
        self.scene_graph = defaultdict(lambda: defaultdict(int))
        point3D_iter = self.point3D_id_to_images.iteritems()

        for i, (point3D_id, images) in enumerate(point3D_iter):
            if not self.point3D_valid(point3D_id):
                continue

            for image_id1, image_id2 in combinations(images[:,0], 2):
                self.scene_graph[image_id1][image_id2] += 1
                self.scene_graph[image_id2][image_id1] += 1
