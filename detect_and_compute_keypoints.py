##########################################################
# Compute keypoints and descriptors:
##########################################################
import os
import cv2
import time
import argparse
import numpy as np
from scipy.spatial.kdtree import KDTree
import colmap_database as cdb

# Parser definition:
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True,
                help='path to input images folder')
ap.add_argument('-d', '--db_path', type=str, default=None,
                help='path for created database file')


def get_input_arguments(debug=False):
    if debug:
        return 'images/', 'images/'
    args = vars(ap.parse_args())
    image_path = args['path']
    database_path = args['db_path'] if args['db_path'] is not None else image_path

    return image_path, database_path

def kdt_nms(kps, descs=None, r=5, k_max=5):
    """
    Use kd-tree to perform local non-maximum suppression of keypoints
    :param kps: keypoints - list of openCV keypoint
    :param descs: keypoint descriptors - list of np.arrays
    :param r: radius of points to query for removal, larger means less output points - int
    :param k_max: - maximum points retreived in single query, larger means less output points - int
    """
    # Sort by score to keep highest score features in each locality:
    neg_responses = [-k_p.response for k_p in kps]
    order = np.argsort(neg_responses)
    kps = np.array(kps)[order].tolist()

    # Create kd-tree for quick NN queries:
    data = np.array([list(k_p.pt) for k_p in kps])
    kd_tree = KDTree(data)

    # Perform NMS using kd-tree, by querying points by score order and removing neighbors from future queries:
    N = len(kps)
    removed = set()
    for ii in range(N):
        if ii in removed:
            continue
        dist, inds = kd_tree.query(data[ii, :], k=k_max, distance_upper_bound=r)
        for jj in inds:
            if jj > ii:
                removed.add(jj)
    kp_filtered = [k_p for k, k_p in enumerate(kps) if k not in removed]
    descs_filtered = None
    if descs is not None:
        descs = descs[order]
        descs_filtered = np.array([desc for k, desc in enumerate(descs) if k not in removed], dtype=np.uint8)
    return kp_filtered, descs_filtered

def compute_descriptors(images_path):
    """
    Computes keypoints and ORB descriptors for a list of images.
    :param images_path: a list of images paths
    :return: kps: detected keypoints for each image - list of opencv keypoint lists
    :return: dsks: computed ORB descriptors for each image - list np.arrays
    :return: img_name_list: list of images name to enable storing in database
    :return: ref_image: the first image in dateset
    """

    # Initiate ORB detector
    #orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, scoreType=cv2.ORB_HARRIS_SCORE, fastThreshold=20)
    #orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, scoreType=cv2.ORB_FAST_SCORE, fastThreshold=20)
    orb = cv2.ORB_create(nfeatures=3000)

    img_name_list = []
    number_of_images = len(image_paths)
    image_index = 0
    kps, dsks = [], []
    reference_image = None
    for file in images_path:
        if file.endswith(('.jpg', '.JPG', '.png')):
            image = cv2.imread(file)
            if image is None:
                continue

            image_index += 1
            if image_index == 1:
                reference_image = image


            image_file_name = os.path.basename(file)
            img_name_list.append(image_file_name)

            # Detect keypoints and compute descriptors for each image:
            keypoint, dsk = orb.detectAndCompute(image, None)
            #keypoint, dsk = kdt_nms(keypoint, dsk)
            """print('Processed file [', image_index, '/', number_of_images, ']')
            print('  Name:            ', image_file_name)
            print('  Dimensions:      ', image.shape[1], ' x ', image.shape[0])
            print('  Features:        ', len(keypoint))"""

            kps.append(keypoint)
            dsks.append(np.tile(dsk, (1, 4)))  # Descriptors are tiled to fit COLMAP's requirements

    return kps, dsks, img_name_list, reference_image

def get_matches_exhaustive(descriptors):
    """
    Exhaustive matching for each pair of descriptors.
    :param descriptors: a list of descriptors - list of np.arrays
    :return: matches: matching feature ids - list np.arrays
    :return: pair_ids: the associated image ids - list of tuples
    """

    """# FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    # index_params = dict(algorithm=FLANN_INDEX_LSH, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    # Create FLANN_Matcher object:
    matcher = cv2.FlannBasedMatcher(index_params, search_params)"""

    # create BFMatcher object
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Exhaustive matching for each pair of images:
    matches, pair_ids = [], []
    for i in range(len(descriptors)):
        start_match = time.time()
        for j in range(i + 1, len(descriptors)):
            # Match current images pair descriptors:
            """k_matches = matcher.knnMatch(descriptors[i], descriptors[j], k=2)
            k_matches = np.array([row for row in k_matches if len(row) == 2])
            
            matches_tmp = []
            for m, n in k_matches:
                print(m.distance, '  ', len(k_matches))
                if m.distance < 0.75 * n.distance:
                    matches_tmp.append([m.queryIdx, m.trainIdx])"""

            match = bf_matcher.match(descriptors[i], descriptors[j])

            # Delete matches between pairs that don't share enough matches:
            if len(match) < 200:
                continue

            # Sort current matches from best to worst:
            match = sorted(match, key=lambda x: x.distance)
            threshold_percent = int(len(match) * 50 / 100)
            matches_tmp = []
            for match_index in range(threshold_percent):
                matches_tmp.append([match[match_index].queryIdx, match[match_index].trainIdx])
            # Remove matches that are too far:
            # 50%
            #matches_tmp = match[0:threshold_percent]
            """for mat in match:
                if mat.distance <= 150:
                    matches_tmp.append([mat.queryIdx, mat.trainIdx])
                else:
                    break"""

            # Add current pair ids and fine matches:
            matches.append(np.array(matches_tmp))
            pair_ids.append((i + 1, j + 1))

        end_match = time.time()
        print('Image {} matching took:'.format(i + 1), end_match - start_match)
    return matches, pair_ids


##########################################################
# Initialize script
##########################################################

# Get input arguments:
image_path, database_path = get_input_arguments()
image_paths = [os.path.join(image_path, file) for file in sorted(os.listdir(image_path))]

##########################################################
# Feature Extraction
##########################################################
start = time.time()
keypoints, descriptors, names, ref_image = compute_descriptors(image_paths)
end = time.time()
fe_time = end - start
print(fe_time, 'seconds for feature extraction')

##########################################################
# Exhaustive matching for each pair of images:
##########################################################
start = time.time()
matches, pair_ids = get_matches_exhaustive(descriptors)
end = time.time()
fm_time = end - start
print(fm_time, 'seconds for feature matching')

##########################################################
# Import to COLMAP format:
##########################################################
start = time.time()
# Open the SQL database:
db = cdb.COLMAPDatabase.connect(os.path.join(database_path, 'database.db'))

# Create all tables upfront:
db.create_tables()

# Add images to database:
_width = ref_image.shape[1]
_height = ref_image.shape[0]
default_focal_length_factor = 1.2
focal_length = max(_width, _height) * default_focal_length_factor

# Camera calibration and distortion parameters (OpenCV)
camera_fx = 942.45
camera_fy = 949.58
camera_cx = 466.81
camera_cy = 324.54
camera_k1 = -0.03
camera_k2 = 0.105
camera_p1 = 0.001
camera_p2 = -0.0006
for i, image_name in enumerate(names):
    # Add a single camera model for all images (for model=2, parameters are (f,cx,cy,k)):
    """camera_id = db.add_camera(model=2, width=_width, height=_height,
                              params=np.array((focal_length, _width / 2, _height / 2, -0.5)))"""
    camera_id = db.add_camera(model=4, width=_width, height=_height,
                              params=np.array((camera_fx,camera_fy,camera_cx,camera_cy,camera_k1,camera_k2,camera_p1,camera_p2)),prior_focal_length=focal_length)
    exec('image_id{} = db.add_image(image_name, camera_id)'.format(i + 1))

# Add keypoints and descriptors to database:
for i, image_name in enumerate(names):
    kp = np.array([[k.pt[0], k.pt[1], k.size / 31, k.angle] for k in keypoints[i]])
    exec('db.add_keypoints(image_id{}, kp)'.format(i + 1))
    exec('db.add_descriptors(image_id{}, descriptors[{}])'.format(i + 1, i))

# Add each pair's matches to database:
for p in range(len(pair_ids)):
    if len(matches[p]) > 0:
        exec('db.add_matches(image_id{}, image_id{}, matches[p])'.format(pair_ids[p][0], pair_ids[p][1]))

# Commit and cleanup the data to the file:
db.commit()
db.close()

end = time.time()
dm_time = end - start
print(dm_time, 'seconds for database prep')

# Create log file:
with open(os.path.join(database_path, 'sfm_log.txt'), 'w') as file:
    file.write('SFM Log File:\n')
    file.write('feature extraction [sec]: {}\n'.format(fe_time))
    file.write('feature matching   [sec]: {}\n'.format(fm_time))
    file.write('database file prep [sec]: {}\n'.format(dm_time))
