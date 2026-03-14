import numpy as np
import os
import sqlite3


#-------------------------------------------------------------------------------
# convert SQLite BLOBs to/from numpy arrays

def array_to_blob(arr):
    return np.getbuffer(arr)

def blob_to_array(blob, dtype, shape=(-1,)):
    return np.frombuffer(blob, dtype).reshape(*shape)


#-------------------------------------------------------------------------------
# convert to/from image pair ids

MAX_IMAGE_ID = 2**31 - 1

def get_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def get_image_ids_from_pair_id(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    return (pair_id - image_id2) / MAX_IMAGE_ID, image_id2


#-------------------------------------------------------------------------------
# create table commands

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))"""

CREATE_INLIER_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([CREATE_CAMERAS_TABLE, CREATE_DESCRIPTORS_TABLE,
    CREATE_IMAGES_TABLE, CREATE_INLIER_MATCHES_TABLE, CREATE_KEYPOINTS_TABLE,
    CREATE_MATCHES_TABLE, CREATE_NAME_INDEX])


#-------------------------------------------------------------------------------
# functional interface for adding objects

def add_camera(db, model, width, height, params, prior_focal_length=False,
               camera_id=None):
    # TODO: Parameter count checks
    params = np.asarray(params, np.float64)
    db.execute("INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
        (camera_id, model, width, height, array_to_blob(params),
         prior_focal_length))


def add_descriptors(db, image_id, descriptors):
    descriptors = np.ascontiguousarray(descriptors, np.uint8)
    db.execute("INSERT INTO descriptors VALUES (?, ?, ?, ?)",
        (image_id,) + descriptors.shape + (array_to_blob(descriptors),))


def add_image(db, name, camera_id, prior_q=np.zeros(4), prior_t=np.zeros(3),
        image_id=None):
    db.execute("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
         prior_q[3], prior_t[0], prior_t[1], prior_t[2]))


# config: defaults to fundamental matrix
def add_inlier_matches(db, image_id1, image_id2, matches, config=2, F=None,
                       E=None, H=None):
    assert(len(matches.shape) == 2)
    assert(matches.shape[1] == 2)

    if image_id1 > image_id2:
        matches = matches[:,::-1]

    if F is not None:
        F = np.asarray(F, np.float64)
    if E is not None:
        E = np.asarray(E, np.float64)
    if H is not None:
        H = np.asarray(H, np.float64)

    pair_id = get_pair_id(image_id1, image_id2)
    matches = np.asarray(matches, np.uint32)
    db.execute("INSERT INTO inlier_matches VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (pair_id,) + matches.shape + (array_to_blob(matches), config, F, E, H))


def add_keypoints(db, image_id, keypoints):
    assert(len(keypoints.shape) == 2)
    assert(keypoints.shape[1] in [2, 4, 6])

    keypoints = np.asarray(keypoints, np.float32)
    db.execute("INSERT INTO keypoints VALUES (?, ?, ?, ?)",
        (image_id,) + keypoints.shape + (array_to_blob(keypoints),))


# config: defaults to fundamental matrix
def add_matches(db, image_id1, image_id2, matches):
    assert(len(matches.shape) == 2)
    assert(matches.shape[1] == 2)

    if image_id1 > image_id2:
        matches = matches[:,::-1]

    pair_id = get_pair_id(image_id1, image_id2)
    matches = np.asarray(matches, np.uint32)
    db.execute("INSERT INTO matches VALUES (?, ?, ?, ?)",
        (pair_id,) + matches.shape + (array_to_blob(matches),))


#-------------------------------------------------------------------------------
# simple functional interface

class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.initialize_tables = lambda: self.executescript(CREATE_ALL)

        self.initialize_cameras = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.initialize_descriptors = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.initialize_images = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.initialize_inlier_matches = \
            lambda: self.executescript(CREATE_INLIER_MATCHES_TABLE)
        self.initialize_keypoints = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.initialize_matches = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)

        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)


    add_camera = add_camera
    add_descriptors = add_descriptors
    add_image = add_image
    add_inlier_matches = add_inlier_matches
    add_keypoints = add_keypoints
    add_matches = add_matches


#-------------------------------------------------------------------------------

def main(args):
    import os

    if os.path.exists(args.database_path):
        print("Error: database path already exists -- will not modify it.")
        exit()

    db = COLMAPDatabase.connect(args.database_path)

    #
    # for convenience, try creating all the tables upfront
    #

    db.initialize_tables()


    #
    # create dummy cameras
    #

    model1, w1, h1, params1 = 0, 1024, 768, np.array((1024., 512., 384.))
    model2, w2, h2, params2 = 2, 1024, 768, np.array((1024., 512., 384., 0.1))

    db.add_camera(model1, w1, h1, params1)
    db.add_camera(model2, w2, h2, params2)


    #
    # create dummy images
    #

    db.add_image("image1.png", 0)
    db.add_image("image2.png", 0)
    db.add_image("image3.png", 2)
    db.add_image("image4.png", 2)


    #
    # create dummy keypoints; note that COLMAP supports 2D keypoints (x, y),
    # 4D keypoints (x, y, theta, scale), and 6D affine keypoints
    # (x, y, a_11, a_12, a_21, a_22)
    #

    N = 1000
    kp1 = np.random.rand(N, 2) * (1024., 768.)
    kp2 = np.random.rand(N, 2) * (1024., 768.)
    kp3 = np.random.rand(N, 2) * (1024., 768.)
    kp4 = np.random.rand(N, 2) * (1024., 768.)

    db.add_keypoints(1, kp1)
    db.add_keypoints(2, kp2)
    db.add_keypoints(3, kp3)
    db.add_keypoints(4, kp4)


    #
    # create dummy matches
    #

    M = 50
    m12 = np.random.randint(N, size=(M, 2))
    m23 = np.random.randint(N, size=(M, 2))
    m34 = np.random.randint(N, size=(M, 2))

    db.add_matches(1, 2, m12)
    db.add_matches(2, 3, m23)
    db.add_matches(3, 4, m34)


    #
    # check cameras
    #

    rows = db.execute("SELECT * FROM cameras")

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float32)
    assert model == model1 and width == w1 and height == h1
    assert np.allclose(params, params1)

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float32)
    assert model == model2 and width == w2 and height == h2
    assert np.allclose(params, params2)


    #
    # check keypoints
    #

    kps = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))

    assert np.allclose(kps[1], kp1)
    assert np.allclose(kps[2], kp2)
    assert np.allclose(kps[3], kp3)
    assert np.allclose(kps[4], kp4)


    #
    # check matches
    #

    pair_ids = [get_pair_id(*pair) for pair in [(1, 2), (2, 3), (3, 4)]]

    matches = dict(
        (get_image_ids_from_pair_id(pair_id),
            blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches"))

    assert np.all(matches[(1, 2)] == m12)
    assert np.all(matches[(2, 3)] == m23)
    assert np.all(matches[(3, 4)] == m34)
    
    #
    # clean up
    #

    db.close()
    os.remove(args.database_path)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--database_path", type=str, default="database.db")

    args = parser.parse_args()

    main(args)
