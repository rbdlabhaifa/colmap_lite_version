import math

import deepdish as dd
import shutil
import subprocess
from os import path, makedirs, remove, listdir
from argparse import ArgumentParser
from threading import Thread
import numpy as np
from matplotlib import pyplot as plt

number_of_images_in_temp_model = 10


def parse_args() -> str:
    """
    Function to parse user argument
    :return: workspace_path
    """
    ap = ArgumentParser(description='Create camera_pose files.')
    ap.add_argument('--workspace_path', required=True)
    args = vars(ap.parse_args())
    return args['workspace_path']


def remove_extra_images(path_to_images: str, number_of_images: int) -> None:
    """
    The function remove all the extra images created in images folder
    :param path_to_images: path to model images folder
    :param number_of_images: the number of image to reconstruct our model (87 by default)
    """
    last_image = 'image' + str(number_of_images) + '.jpg'
    while last_image in listdir(path_to_images):
        last_image_path = path.join(path_to_images, last_image)
        remove(last_image_path)
        print(f"remove {last_image}")
        number_of_images += 1
        last_image = 'image' + str(number_of_images) + '.jpg'


def prepare_video(path_to_video: str, number_of_images=110) -> None:
    """
    The function prepare the images for our model based on given video
    :param path_to_video: video in h264 format
    :param number_of_images: the number of image to reconstruct our model (110 by default)
    """

    temp_video = path.join(path_to_video, 'temp_outpy.mp4')
    video = path.join(path_to_video, 'outpy.h264')

    # create mp4 video for metadata and compute video duration
    subprocess.run(['ffmpeg', '-i', video, '-c', 'copy', temp_video])
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", temp_video],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    video_duration = float(result.stdout)

    # create images folder
    path_to_images = path.join(path_to_video, 'images')
    if path.exists(path_to_images) and path.isdir(path_to_images):
        shutil.rmtree(path_to_images)
    makedirs(path_to_images)

    # split the given video into images
    subprocess.run(['ffmpeg', '-i', temp_video, '-r', str(number_of_images / video_duration), '-f', 'image2',
                    path.join(path_to_images, 'image%d.jpg')])

    # remove extra files
    remove_extra_images(path_to_images, number_of_images)
    remove(temp_video)


def create_temp_model(temp_dir_path: str) -> str:
    """
    The function prepare the images for our model based on given video
    :param temp_dir_path: video in h264 format
    :return number_of_images: path to temporary model folder
    """

    # create temp images folder
    path_to_temp_model = path.join(temp_dir_path, 'temp_model')
    path_to_temp_images = path.join(path_to_temp_model, 'temp_images')

    # remove old temporary folder if exists
    if path.exists(path_to_temp_model) and path.isdir(path_to_temp_model):
        shutil.rmtree(path_to_temp_model)

    number_of_temp_images = 0
    path_to_images = path.join(temp_dir_path, 'images')

    # take only part of the images for the temp model
    while number_of_temp_images < number_of_images_in_temp_model:
        try:
            number_of_temp_images = len([name for name in listdir(path_to_images) if name.endswith('.jpg')])
        except FileNotFoundError:
            number_of_temp_images = 0

    # copy subdirectory example
    shutil.copytree(path_to_images, path_to_temp_images)

    # run colmap to create model for the first 10 images in video
    subprocess.run(['colmap', 'automatic_reconstructor',
                    '--workspace_path', path_to_temp_model, '--image_path', path_to_temp_images,
                    '--data_type=video', '--quality=extreme'])

    return path_to_temp_model


def quaternion_to_rotation_matrix(q0, q1, q2, q3) -> np:
    """
    The function convert quaternion vector to rotation matrix
    https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    :param q0: the value of qw
    :param q1: the value of qx
    :param q2: the value of qy
    :param q3: the value of qz
    :return rot_matrix: rotation matrix 3x3 as numpy array
    """

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def rotation_matrix_to_quaternion(rotation_matrix: np) -> object:
    """
    The function convert rotation matrix to quaternion vector
    https://learnopencv.com/rotation-matrix-to-euler-angles/
    :param rotation_matrix: rotation matrix 3x3 represented by numpy array
    :return quaternion vector: represented by (qx, qy, qz, qw)
    """

    cosine_for_pitch = math.sqrt(rotation_matrix[0][0] ** 2 + rotation_matrix[1][0] ** 2)
    is_singular = cosine_for_pitch < 10 ** -6
    if not is_singular:
        yaw = math.atan2(rotation_matrix[1][0], rotation_matrix[0][0])
        pitch = math.atan2(-rotation_matrix[2][0], cosine_for_pitch)
        roll = math.atan2(rotation_matrix[2][1], rotation_matrix[2][2])
    else:
        yaw = math.atan2(-rotation_matrix[1][2], rotation_matrix[1][1])
        pitch = math.atan2(-rotation_matrix[2][0], cosine_for_pitch)
        roll = 0

    e = (yaw, pitch, roll)

    return euler_to_quaternion(e)


def euler_to_quaternion(euler: tuple) -> object:
    """
    The function convert euler angle to quaternion object
    :param euler: angle represented by yaw, pitch, roll
    :return quaternion vector: represented by (qx, qy, qz, qw)
    """
    (yaw, pitch, roll) = (euler[0], euler[1], euler[2])
    qy = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qx = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return qx, qy, qz, qw


def get_first_image_pose(image_src: str) -> list:
    """
    The function return the absolut R & T for the first image in temp model
    :param image_src: path to image file (colmap output)
    :return R&T: R = list[0], T list[1] or None if image1 not exists
    """
    # read images file
    with open(image_src, 'r') as file:
        lines = file.readlines()[4::2]

    # create absolut camera pose dictionary
    for line in lines:
        columns = line.split()
        image_name = columns[9].split('.')[0]
        image_id = int(image_name.split('e')[1])

        # convert and return the camera pose for the first image in model
        if image_id == 1:
            qw = float(columns[1])
            qx = float(columns[2])
            qy = float(columns[4])
            qz = float(columns[3])
            rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            tx = float(columns[5])
            ty = float(columns[7])
            tz = float(columns[6])
            translation_vector = np.array([tx, ty, tz])
            return [rotation_matrix, translation_vector]

    return []


def draw_rel_camera_pose(image: int, origin: list, camera_pose: list, plot_dir_path: str) -> None:
    """
    Debug function for plotting the relative camera poses
    :param image: number of current image
    :param origin: list of [x,y,z] of the origin
    :param camera_pose: list of [x1,y1,z1][x2,y2,z2] of the camera pose (three 2d vectors)
    :param plot_dir_path: path to plot directory
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10.)
    ax.set_title('camera pose image: %d' % image)
    scale = 7
    ax.set_xlim3d(-scale, scale)
    ax.set_ylim3d(-scale, scale)
    ax.set_zlim3d(-scale, scale)

    # replace the Y-Axis with Z-Axis
    ax.scatter(origin[0], origin[2], origin[1], c='black')
    for i in range(3):
        ax.plot([origin[0], camera_pose[i][0]], [origin[2], camera_pose[i][2]], [origin[1], camera_pose[i][1]])
        i += 1

    fig.savefig(f'{plot_dir_path}/%d.png' % image)
    plt.close(fig)
    plt.clf()


def compute_absolut_camera_pose(camera_pose_rel_dict: dict, first_image_pose: list,
                                workspace_path: str, do_plot=False) -> dict:
    """
    The function return a dictionary with recovered R&T for each image
    :param camera_pose_rel_dict: dictionary of relative camera poses for each image
    :param first_image_pose: absolute R&T of the first image
    :param workspace_path: path to workspace_path
    :param do_plot: boolean flag for debug purpose
    :return: camera_pose_recover dictionary
    """

    # create directory for reference plots
    ref_pose_images_path = path.join(workspace_path, 'ref_images')
    if do_plot:
        makedirs(ref_pose_images_path)

    # initialize parameters for computing absolut camera poses
    camera_pose_recover = {}
    rotation = first_image_pose[0]
    translation = first_image_pose[1]

    is_first = True

    prev_rotation = np.identity(3)
    prev_translation = np.zeros(3)

    # foreach image compute the absolut pose out of the reference pose
    for image in camera_pose_rel_dict.keys():
        rel_rotation = camera_pose_rel_dict[image][0]
        rel_translation = camera_pose_rel_dict[image][1]

        # for the first image, take the values from the temporary model
        if not is_first:
            rotation = rel_rotation @ np.linalg.inv(prev_rotation.T)
            translation = rel_translation + prev_translation

        # compute the absolut camera pose
        camera_pose = rotation + translation

        if do_plot:
            draw_rel_camera_pose(image, translation, camera_pose, ref_pose_images_path)

        # save the values foreach image (in R & T format)
        camera_pose_recover[image] = [rotation, translation]

        prev_rotation = rotation
        prev_translation = translation
        is_first = False

    return camera_pose_recover


def write_camera_pose_to_file(camera_pose_abs_dict: dict, pose_dir_path: str) -> None:
    """
    The function write the recovered camera poses according to COLMAP documentation
    :param camera_pose_abs_dict: dictionary of recovered camera poses for each image
    :param pose_dir_path: path to image file
    """
    image_dst = path.join(pose_dir_path, 'images.txt')
    with open(image_dst, 'w+') as file:
        file.write('# Image list with two lines of data per image:\n')
        file.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        file.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        file.write(f'# Number of images: {len(camera_pose_abs_dict.keys())}\n')

        # write each camera pose to file
        for image in camera_pose_abs_dict.keys():
            image_pose_data = []
            t_vector = camera_pose_abs_dict[image][1]
            qx, qy, qz, qw = rotation_matrix_to_quaternion(camera_pose_abs_dict[image][0])

            image_pose_data.append(str(image))
            # image_pose_data.append(f'{qw} {qx} {qy} {qz}')
            image_pose_data.append(f'{qz} {qy} {qx} {qw}')
            image_pose_data.append(' '.join(map(str, t_vector)))
            image_pose_data.append('1')
            image_pose_data.append(f'image{image}.jpg')

            file.write(' '.join(image_pose_data) + '\n\n')


def main():
    # Parse input arguments:
    workspace_path = parse_args()

    # make sure the workspace in empty
    for filename in listdir(workspace_path):
        if filename.endswith('.h264'):
            continue
        path_to_node = path.join(workspace_path, filename)
        if path.isdir(path_to_node):
            shutil.rmtree(path_to_node)
        else:
            remove(path_to_node)

    # prepare video and create the images for our model
    video_thread = Thread(target=prepare_video, args=(workspace_path, 87))
    video_thread.start()

    # create temp folder for temp model
    temp_model_workspace_path = create_temp_model(workspace_path)

    # create camera pose parameters
    pose_output_path = path.join(workspace_path, 'camera_poses')
    makedirs(pose_output_path)

    # create camera input file
    camera_src = path.join(temp_model_workspace_path, 'sparse/0/cameras.txt')
    camera_dst = path.join(pose_output_path, 'cameras.txt')
    shutil.copyfile(camera_src, camera_dst)

    # create an empty points input file
    points_dst = path.join(pose_output_path, 'points3D.txt')
    open(points_dst, 'w').close()

    # get camera poses for first image
    image_src = path.join(temp_model_workspace_path, 'sparse/0/images.txt')
    first_image_pose = get_first_image_pose(image_src)

    if not first_image_pose:
        print("Error in temp model - cant compute the camera pose for the first image")
        exit(1)

    # reading the reference pose model from file
    camera_pose_rel_dict = dd.io.load('ref_camera_pose.h5')
    camera_pose_abs_dict = compute_absolut_camera_pose(camera_pose_rel_dict, first_image_pose, workspace_path,
                                                       do_plot=True)

    # create the image file according to COLMAP documentation
    write_camera_pose_to_file(camera_pose_abs_dict, pose_output_path)

    # wait for the video thread before closing the process
    video_thread.join()


if __name__ == "__main__":
    print('==============================================================================')
    print('workplace preparation')
    print('==============================================================================')

    main()
