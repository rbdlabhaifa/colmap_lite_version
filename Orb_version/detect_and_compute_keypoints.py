import os
from multiprocessing import cpu_count, Pool
import copyreg
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


def get_input_arguments() -> tuple:
    """
    Function to parse user argument
    :return: image_path, database_path
    """
    args = vars(ap.parse_args())
    image_path = args['path']
    database_path = args['db_path'] if args['db_path'] is not None else image_path

    return image_path, database_path


def compute_descriptors(images: list) -> tuple:
    """
    Computes keypoints and ORB descriptors for a list of images.
    :param images: a list of images paths
    :return: keypoints: detected keypoints for each image - list of opencv keypoint lists
    :return: descriptors: computed ORB descriptors for each image - list np.arrays
    :return: img_name_list: list of images name to enable storing in database
    """
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=3000, scoreType=cv2.ORB_FAST_SCORE)

    img_name_list, keypoints, descriptors = [], [], []

    for file in images:
        if file.endswith(('.jpg', '.JPG', '.png')):
            image = cv2.imread(file)

            if image is None:
                continue

            image_file_name = os.path.basename(file)
            img_name_list.append(image_file_name)

            # Detect keypoints and compute descriptors for each image:
            keypoint, dsk = orb.detectAndCompute(image, None)

            keypoints.append(keypoint)
            descriptors.append(np.tile(dsk, (1, 4)))  # Descriptors are tiled to fit COLMAP requirements

    return keypoints, descriptors, img_name_list


def filter_BF_matches(matches: list, threshold=45) -> list:
    """
    filter matches list and keep the best matches according to threshold
    :param matches: a list of matches
    :param threshold: threshold filtering
    :return: matches_tmp: list of the best matches
    """
    matches_tmp = []
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    threshold_percent = int(len(sorted_matches) * threshold / 100)
    for match_index in range(threshold_percent):
        matches_tmp.append([sorted_matches[match_index].queryIdx, sorted_matches[match_index].trainIdx])

    return matches_tmp


def get_matches_sequential(descriptors: list, match_window_overlap=5, do_quadratic_match=False) -> tuple:
    """
    sequential matching according to match_window_overlap.
    :param descriptors: a list of descriptors - list of np.arrays
    :param match_window_overlap: the size of the match window of each image
    :param do_quadratic_match: boolean flag for quadratic matching
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

            if do_quadratic_match:
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

    # loop detection
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


def patch_Keypoint_pickling() -> None:
    """
    Allows multi-processing with open-cv's Keypoints, by making them pickle-able.
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


def chunk(in_list: list, num_item: int) -> list:
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


def update_camera_pose_file(workspace_path: str, images_id_in_db: dict) -> None:
    """
    update the images.txt file according to the image table in workspace DB
    :param workspace_path:  path to workspace folder
    :param images_id_in_db: dictionary of all the images and there indexes in DB
    :return: separate sub-item lists
    """
    camera_pose_path = os.path.join(workspace_path, 'camera_poses/images.txt')
    if os.path.exists(camera_pose_path):
        val_list = list(images_id_in_db.values())

        # read image.txt file
        f = open(camera_pose_path, "r")
        lines = f.readlines()[4::2]
        f.close()

        # update the file according to DB
        with open(camera_pose_path, 'w') as file:
            for line in lines:
                data = line.split()
                data[0] = str(val_list.index(data[9]) + 1)
                file.write(' '.join(data) + '\n\n')


def main():
    ##########################################################
    # Initialize script
    ##########################################################
    image_path, database_path = get_input_arguments()
    images = [os.path.join(image_path, file) for file in sorted(os.listdir(image_path))]

    # Patch open-cv's keypoint class:
    patch_Keypoint_pickling()

    # Get number of available processes and number of images per process:
    cpu_num = cpu_count()
    img_per_process = int(np.ceil(len(images) / float(cpu_num)))

    # Initialize a Pool object:
    pool = Pool(processes=cpu_num)

    print('\n==============================================================================')
    print('Feature extraction')
    print('==============================================================================\n')
    start = time.time()

    # compute descriptors with multiprocessing
    chunked_images = list(chunk(images, img_per_process))
    temp = pool.map(compute_descriptors, chunked_images)
    keypoints = [item for sublist in temp for item in sublist[0]]
    descriptors = [item for sublist in temp for item in sublist[1]]
    names = [item for sublist in temp for item in sublist[2]]

    end = time.time()
    fe_time = (end - start)
    print('Elapsed time: %.2f [seconds]' % fe_time)

    print('\n==============================================================================')
    print('Sequential matching')
    print('==============================================================================\n')

    start = time.time()

    # get matches with multiprocessing
    matches, pair_ids = get_matches_sequential(descriptors)

    end = time.time()
    fm_time = (end - start)
    print('Elapsed time: %.2f [seconds]' % fm_time)

    ##########################################################
    # Import to COLMAP format:
    ##########################################################
    start = time.time()

    # Open the SQL database:
    db = cdb.COLMAPDatabase.connect(os.path.join(database_path, 'database.db'))

    # Create all tables upfront:
    db.create_tables()

    """
    camera matrix:
        [505.61918164   0.         314.90808858]
        [  0.         506.35795295 233.38488026]
        [  0.           0.           1.        ]
        
    distortion coefficients:  
        [ 0.10637077  0.41155632 -0.00463533 -0.00815625 -2.5849433 ]
        
        https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    """
    fx = 505.61918164
    fy = 506.35795295
    cx = 314.90808858
    cy = 233.38488026
    k1 = 0.10637077
    k2 = 0.41155632
    p1 = -0.00463533
    p2 = -0.00815625

    # Add a single camera model for all images (for model=4, parameters are (fx, fy, cx, cy, k1, k2, p1, p2)):
    camera_id = db.add_camera(model=4, width=640, height=480,
                              params=np.array((fx, fy, cx, cy, k1, k2, p1, p2)))

    # Add images to database:
    images_id_in_db = {}
    for i, image_name in enumerate(names):
        exec('image_id{} = db.add_image(image_name, camera_id)'.format(i + 1))
        images_id_in_db[i + 1] = image_name

    # update camera_pose with id's from DB
    update_camera_pose_file(database_path, images_id_in_db)

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
    fd_time = (end - start)

    # Create log file:
    with open(os.path.join(database_path, 'sfm_log.txt'), "a") as file:
        file.write('feature extraction [sec]: %.2f\n' % fe_time)
        file.write('feature matching   [sec]: %.2f\n' % fm_time)
        file.write('database preparation   [sec]: %.2f\n' % fd_time)


if __name__ == "__main__":
    main()
