###################
# 0. General Setup:
###################

import sqlite3
import numpy as np

MAX_IMAGE_ID = 2**31 - 1

# Strings of SQL Commands:
##########################

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """DROP TABLE IF EXISTS descriptors; CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """DROP TABLE IF EXISTS images; CREATE TABLE IF NOT EXISTS images (
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
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
DROP TABLE IF EXISTS two_view_geometries; CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """DROP TABLE IF EXISTS keypoints; CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """DROP TABLE IF EXISTS matches; CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = '; '.join([CREATE_CAMERAS_TABLE,
                        CREATE_IMAGES_TABLE,
                        CREATE_KEYPOINTS_TABLE,
                        CREATE_DESCRIPTORS_TABLE,
                        CREATE_MATCHES_TABLE,
                        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
                        CREATE_NAME_INDEX])

# Custom Functions:
###################


def image_ids_to_pair_id(image_id1, image_id2):
    """
    Converts two image ids into a single unique pair id.
    :param image_id1: image id 1 - int
    :param image_id2: image id 2 - int
    :return pair_id: a unique pair id - int
    """
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    """
    Converts a single pair id into two image ids.
    :param pair_id: image pair id - int
    :return image_id1: image id 1 - int
    :return image_id2: image id 2 - int
    """
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    """
    Converts an array of numbers into raw data bytes.
    :param array: an array of numbers - np.array
    :return blob: a bytes representation of the array - bytes
    """
    return array.tostring()


def blob_to_array(blob, dtype, shape=(-1,)):
    """
    Converts a raw data bytes representation into an array.
    :param blob: row data bytes representation - bytes
    :param dtype: data type for the array - str
    :param shape: data shape for the array - tuple
    :return array: an array of numbers - np.array
    """
    return np.fromstring(blob, dtype=dtype).reshape(*shape)

########################################################################################################################

###########################
# 1. COLMAP Database Class:
###########################

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        """
        Initializes an SQL file of a COLMAP database format at input path.
        All database tables are created upfront (all are empty).
        """
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        """
        Add camera model to database.
        :param model: camera model to use out of [0-SIMPLE_PINHOLE, 1-PINHOLE, 2-SIMPLE_RADIAL...] - int
        :param width: frame width for current camera - int
        :param height: frame height for current camera - int
        :param params: camera parameters (amount depends on model) - tuple
        :param prior_focal_length: known camera focal length - float
        :param camera_id: unique camera id, if not specified, a new id is created - int
        :return: camera_id: the unique camera id - int
        """
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        """
        Add camera model to database.
        :param name: path to image in image folder - str
        :param camera_id: the unique camera id - int
        :param prior_q: image's camera coordinate system quaternion (Qw, Qx, Qy, Qz) - np.array
        :param prior_t: image's camera coordinate system translation vector (Tx, Ty, Tz) - np.array
        :param image_id: unique image id, if not specified, a new id is created - int
        :return: image_id: the unique image id - int
        """
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        """
        Add keypoints to an image.
        :param image_id: the unique image id - int
        :param keypoints: a list of keypoints, each is [x,y,scale,angle] - list of lists
        """
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        """
        Add keypoint descriptors to an image.
        :param image_id: the unique image id - int
        :param descriptors: a list of descriptors, each of length 128 - list of np.arrays
        """
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        """
        Add keypoints matches between a pair of images.
        :param image_id1: image id 1 - int
        :param image_id2: image id 2 - int
        :param matches: an array of feature matches of size (n,2) - np.array
        """
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        """
        Add known geometrical transformation between a pair of images.
        :param image_id1: image id 1 - int
        :param image_id2: image id 2 - int
        :param matches: an array of feature matches of size (n,2) - np.array
        :param F: Fundamental matrix of size (3,3) - np.array
        :param E: Essential matrix of size (3,3) - np.array
        :param H: Homography matrix of size (3,3) - np.array
        :param config: configuration of two-view geometry [1-DEGENERATE, 2-CALIBRATED, 3-UNCALIBRATED...] - int
        """
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))


########################################################################################################################

###################
# 2. Example Usage:
###################

def example_usage():
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()
    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    # Open the database:
    db = COLMAPDatabase.connect(args.database_path)
    # For convenience, create all tables upfront:
    db.create_tables()

    # Create dummy cameras:
    model1, width1, height1, params1 = 0, 1024, 768, np.array((1024., 512., 384.))
    model2, width2, height2, params2 = 2, 1024, 768, np.array((1024., 512., 384., 0.1))
    camera_id1 = db.add_camera(model1, width1, height1, params1)
    camera_id2 = db.add_camera(model2, width2, height2, params2)

    # Create dummy images:
    image_id1 = db.add_image("image1.png", camera_id1)
    image_id2 = db.add_image("image2.png", camera_id1)
    image_id3 = db.add_image("image3.png", camera_id2)
    image_id4 = db.add_image("image4.png", camera_id2)

    # Create dummy keypoints.
    # Note that COLMAP supports:
    #      - 2D keypoints: (x, y)
    #      - 4D keypoints: (x, y, scale, angle)
    #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)
    num_keypoints = 1000
    keypoints1 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints2 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints3 = np.random.rand(num_keypoints, 2) * (width2, height2)
    keypoints4 = np.random.rand(num_keypoints, 2) * (width2, height2)
    db.add_keypoints(image_id1, keypoints1)
    db.add_keypoints(image_id2, keypoints2)
    db.add_keypoints(image_id3, keypoints3)
    db.add_keypoints(image_id4, keypoints4)

    # Create dummy matches:
    M = 50
    matches12 = np.random.randint(num_keypoints, size=(M, 2))
    matches23 = np.random.randint(num_keypoints, size=(M, 2))
    matches34 = np.random.randint(num_keypoints, size=(M, 2))
    db.add_matches(image_id1, image_id2, matches12)
    db.add_matches(image_id2, image_id3, matches23)
    db.add_matches(image_id3, image_id4, matches34)

    # Commit the data to the file:
    db.commit()
    # Read and check cameras:
    rows = db.execute("SELECT * FROM cameras")

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id1
    assert model == model1 and width == width1 and height == height1
    assert np.allclose(params, params1)

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id2
    assert model == model2 and width == width2 and height == height2
    assert np.allclose(params, params2)

    # Read and check keypoints:
    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))

    assert np.allclose(keypoints[image_id1], keypoints1)
    assert np.allclose(keypoints[image_id2], keypoints2)
    assert np.allclose(keypoints[image_id3], keypoints3)
    assert np.allclose(keypoints[image_id4], keypoints4)

    # Read and check matches:
    pair_ids = [image_ids_to_pair_id(*pair) for pair in
                ((image_id1, image_id2),
                 (image_id2, image_id3),
                 (image_id3, image_id4))]
    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches"))
    assert np.all(matches[(image_id1, image_id2)] == matches12)
    assert np.all(matches[(image_id2, image_id3)] == matches23)
    assert np.all(matches[(image_id3, image_id4)] == matches34)

    # Clean up:
    db.close()
    if os.path.exists(args.database_path):
        os.remove(args.database_path)


if __name__ == "__main__":
    example_usage()
