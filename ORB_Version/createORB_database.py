###################
# 0. General Setup:
###################

import os
import cv2
import time
import copyreg
import argparse
import numpy as np
import colmap_database as cdb
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy.spatial.kdtree import KDTree

# Parser definition:
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True,
                help='path to input images folder')
ap.add_argument('-d', '--db_path', type=str, default=None,
                help='path for created database file')


# Custom Functions:
###################

def patch_Keypoint_pickling():
    """
    Allows multi-processing with open-cv's Keypoints, by making them picklable.
    Creates the bundling between class and arguments.
    """
    def _pickle_keypoint(keypoint):  # cv2.KeyPoint
        return cv2.KeyPoint, (
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            keypoint.angle,
            keypoint.response,
            keypoint.octave,
            keypoint.class_id,
        )
    # Apply the bundling to pickle:
    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoint)


def chunk(in_list, num_item):
    """
    Used for multi-processing.
    :param in_list:  input item for the function running in multi-process - list
    :param num_item: number of items for each sub-item used in a single process - int
    :return: separate sub-item lists
    """
    # loop over the list in n-sized chunks
    for idx in range(0, len(in_list), num_item):
        # yield the current n-sized chunk to the calling function
        yield in_list[idx: idx + num_item]


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
    print('Filtered', len(kp_filtered), 'of', N)
    return kp_filtered, descs_filtered


def load_images_from_paths(paths):
    """
    Loads all jpg and png images from folder.
    :param   paths:     paths to image's - list of str
    :return: img_list:  images from the folder - list of np.arrays
    :return: name_list: file names of images inside folder - list of str
    """
    img_list = []
    name_list = []
    for file in paths:
        if file.endswith(('.jpg', '.png')):
            img = cv2.imread(file)
            if img is not None:
                img_list.append(img)
                name_list.append(os.path.basename(file))
    return img_list, name_list


def compute_descriptors(images):
    """
    Computes keypoints and ORB descriptors for a list of images.
    :param images: a list of images - list of np.arrays
    :return: kps: detected keypoints for each image - list of opencv keypoint lists
    :return: dsks: computed ORB descriptors for each image - list np.arrays
    """
    # Initialize ORB detector:
    orb = cv2.ORB_create(nfeatures=20000, scoreType=cv2.ORB_FAST_SCORE, fastThreshold=15)
    # Detect keypoints and compute descriptors for each image:
    kps, dsks = [], []
    for image in images:
        kp, dsk = orb.detectAndCompute(image, None)
        kp, dsk = kdt_nms(kp, dsk)
        kps.append(kp)
        dsks.append(np.tile(dsk, (1, 4)))  # Descriptors are tiled to fit COLMAP's requirements
    return kps, dsks


def get_matches(descriptors):
    """
    Exhaustive matching for each pair of descriptors.
    :param descriptors: a list of descriptors - list of np.arrays
    :return: matches: matching feature ids - list np.arrays
    :return: pair_ids: the associated image ids - list of tuples
    """
    # Create BFMatcher object:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Exhaustive matching for each pair of images:
    matches, pair_ids = [], []
    len_of_desc = len(descriptors)
    match_window = 10
    for i in range(len_of_desc):
        start_match = time.time()
        for j in range(i+1, len_of_desc):
            matches_tmp = []
            if ((j-i)%len_of_desc < match_window) or ((i+j) % len_of_desc < match_window):
                # Match current images pair descriptors:
                match = matcher.match(descriptors[i], descriptors[j])
                # Sort current matches from best to worst:
                match = sorted(match, key=lambda x: x.distance)
                # Remove matches that are too far:

                for mat in match:
                    if mat.distance <= 200:
                        matches_tmp.append([mat.queryIdx, mat.trainIdx])
                    else:
                        break
                # Delete matches between pairs that don't share enough matches:
                if len(matches_tmp) < 200:
                    matches_tmp = matches_tmp[-2:]

            matches.append(np.array(matches_tmp))
            # Add current pair ids:
            pair_ids.append((i+1, j+1))
        end_match = time.time()
        print('Image {}:'.format(i), end_match-start_match)
    return matches, pair_ids


########################################################################################################################

##################################################################
# 1. ORB Features Detection, Matching and Import to COLMAP format:
##################################################################

if __name__ == "__main__":

    # Get input arguments:
    args = vars(ap.parse_args())
    image_path = args['path']
    database_path = args['db_path'] if args['db_path'] is not None else image_path
    #image_path = 'images/'
    #database_path = 'images/'
    image_paths = [os.path.join(image_path, file) for file in sorted(os.listdir(image_path))]

    # Prepare multi-processing tools:
    #################################

    # Patch open-cv's keypoint class:
    patch_Keypoint_pickling()
    # Get number of available processes and number of images per process:
    cpu_num = cpu_count()
    img_per_process = int(np.ceil(len(image_paths) / float(cpu_num)))
    # Initialize a Pool object:
    pool = Pool(processes=cpu_num)

    # Compute features and matches:
    ###############################

    # Load images from folder:
    chunked_paths = list(chunk(image_paths, img_per_process))
    temp = pool.map(load_images_from_paths, chunked_paths)
    images = [item for sublist in temp for item in sublist[0]]
    names = [item for sublist in temp for item in sublist[1]]

    # Compute keypoints and descriptors:
    start = time.time()
    chunked_images = list(chunk(images, img_per_process))
    temp = pool.map(compute_descriptors, chunked_images)
    keypoints = [item for sublist in temp for item in sublist[0]]
    descriptors = [item for sublist in temp for item in sublist[1]]
    end = time.time()
    fe_time = end-start
    print(fe_time, 'seconds for feature extraction')

    # Exhaustive matching for each pair of images:
    start = time.time()
    matches, pair_ids = get_matches(descriptors)
    end = time.time()
    fm_time = end-start
    print(fm_time, 'seconds for feature matching')

    # Import to COLMAP format:
    ##########################

    # Open the SQL database:
    start = time.time()
    db = cdb.COLMAPDatabase.connect(os.path.join(database_path, 'database.db'))
    # Create all tables upfront:
    db.create_tables()
    # Add a single camera model for all images (for model=2, parameters are (f,cx,cy,k)):
    camera_id1 = db.add_camera(model=2, width=images[0].shape[1], height=images[0].shape[0],
                               params=np.array((2304., images[0].shape[1]/2, images[0].shape[0]/2, 0)))
    # Add images to database:
    for i, image_name in enumerate(names):
        exec('image_id{} = db.add_image(image_name, camera_id1)'.format(i+1))
    # Add keypoints and descriptors to database:
    for i, image_name in enumerate(names):
        kp = np.array([[k.pt[0], k.pt[1], k.size/31, k.angle] for k in keypoints[i]])
        exec('db.add_keypoints(image_id{}, kp)'.format(i+1))
        exec('db.add_descriptors(image_id{}, descriptors[{}])'.format(i+1, i))
    # Add each pair's matches to database:
    for p in range(len(pair_ids)):
        if len(matches[p]) > 0:
            exec('db.add_matches(image_id{}, image_id{}, matches[p])'.format(pair_ids[p][0], pair_ids[p][1]))
    # Commit the data to the file:
    db.commit()
    # Clean up:
    db.close()
    end = time.time()
    dm_time = end-start
    print(dm_time, 'seconds for database prep')

    # Create log file:
    with open(os.path.join(database_path, 'sfm_log.txt'), 'w') as file:
        file.write('SFM Log File:\n')
        file.write('feature extraction [sec]: {}\n'.format(fe_time))
        file.write('feature matching   [sec]: {}\n'.format(fm_time))
        file.write('database file prep [sec]: {}\n'.format(dm_time))
