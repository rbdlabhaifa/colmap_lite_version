import os
import cv2
import time
import argparse
import numpy as np
import colmap_database as cdb

# Parser definition:
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True,
                help='path to input images folder')
ap.add_argument('-d', '--db_path', type=str, default=None,
                help='path for created database file')


def get_input_arguments():
    args = vars(ap.parse_args())
    image_path = args['path']
    database_path = args['db_path'] if args['db_path'] is not None else image_path

    return image_path, database_path


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
    orb = cv2.ORB_create(nfeatures=3000)

    img_name_list = []
    number_of_images = len(image_paths)
    kps, dsks = [], []
    reference_image = None
    for file in images_path:
        if file.endswith(('.jpg', '.JPG', '.png')):
            image = cv2.imread(file)
            if image is None:
                continue

            if reference_image is None:
                reference_image = image

            image_file_name = os.path.basename(file)
            img_name_list.append(image_file_name)

            # Detect keypoints and compute descriptors for each image:
            keypoint, dsk = orb.detectAndCompute(image, None)
            kps.append(keypoint)
            dsks.append(np.tile(dsk, (1, 4)))  # Descriptors are tiled to fit COLMAP's requirements

    return kps, dsks, img_name_list, reference_image


def filter_BF_matches(matches, threshold=45):
    matches_tmp = []
    sorted_mathces = sorted(matches, key=lambda x: x.distance)
    threshold_percent = int(len(sorted_mathces) * threshold / 100)
    for match_index in range(threshold_percent):
        matches_tmp.append([sorted_mathces[match_index].queryIdx, sorted_mathces[match_index].trainIdx])

    return matches_tmp

def get_matches_sequential(descriptors, match_window_overlap=10):
    """
    sequential matching according to match_window_overlap.
    :param descriptors: a list of descriptors - list of np.arrays
    :param match_window_overlap: the size of the match window of each image
    :return: matches: matching feature ids - list np.arrays
    :return: pair_ids: the associated image ids - list of tuples
    """
    
    # create BFMatcher object
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches, pair_ids = [], []
    number_of_descriptors = len(descriptors)
    for image_idx1 in range(number_of_descriptors):
        for window in range(1, match_window_overlap):
            image_idx2 = image_idx1 + window

            # Check for overlap
            if image_idx2 >= number_of_descriptors:
                break

            # Match current images pair descriptors:
            current_matches = matcher.match(descriptors[image_idx1], descriptors[image_idx2])
            current_matches = filter_BF_matches(current_matches)
            
            # Add current pair ids and fine matches:
            matches.append(np.array(current_matches))
            pair_ids.append((image_idx1 + 1, image_idx2 + 1))

            image_idx2_quadratic = image_idx1 + (1 << (window - 1))
            # Check for quadratic overlap
            if (image_idx2_quadratic > image_idx1 + match_window_overlap) and (
                    image_idx2_quadratic < number_of_descriptors):
                # Match current images pair descriptors:
                current_matches = matcher.match(descriptors[image_idx1], descriptors[image_idx2_quadratic])
                current_matches = filter_BF_matches(current_matches, 30)
                
                # Add current pair ids and fine matches:
                matches.append(np.array(current_matches))
                pair_ids.append((image_idx1 + 1, image_idx2_quadratic + 1))

    for image_idx1 in range(match_window_overlap):
        image_idx2 = number_of_descriptors + image_idx1 - match_window_overlap

        for loop_closer_image_idx in range(image_idx2, match_window_overlap):
            if (image_idx1 + 1, loop_closer_image_idx + 1) in pair_ids:
                continue

            # Match current images pair descriptors:
            current_matches = matcher.match(descriptors[image_idx1], descriptors[image_idx2])
            current_matches = filter_BF_matches(current_matches)

            # Add current pair ids and fine matches:
            matches.append(np.array(current_matches))
            pair_ids.append((image_idx1 + 1, image_idx2 + 1))

    return matches, pair_ids


##########################################################
# Initialize script
##########################################################
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
matches, pair_ids = get_matches_sequential(descriptors)
end = time.time()
fm_time = end - start
print(fm_time, 'seconds for feature matching')

##########################################################
# Import to COLMAP format:
##########################################################

# Open the SQL database:
db = cdb.COLMAPDatabase.connect(os.path.join(database_path, 'database.db'))

# Create all tables upfront:
db.create_tables()

# No need to import camera parameters
for i, image_name in enumerate(names):
    exec('image_id{} = db.add_image(image_name, 1)'.format(i + 1))

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

# Create log file:
with open(os.path.join(database_path, 'sfm_log.txt'), "w+") as file:
    file.write('SFM Log File:\n')
    file.write('feature extraction [sec]: {}\n'.format(fe_time))
    file.write('feature matching   [sec]: {}\n'.format(fm_time))


