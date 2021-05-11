import collections
import subprocess
import webbrowser
from argparse import ArgumentParser
from os import path, makedirs
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from workplace_preparation import prepare_video, clear_workspace, quaternion_to_rotation_matrix, draw_rel_camera_pose
import deepdish as dd


def plot_model_2d(sparse_folder: str, output_path: str, threshold=500) -> None:
    """
    get a list of 3d points from the points.txt file
    :param sparse_folder: path to points.txt file
    :param output_path: path to save the plot
    :param threshold: threshold for filter outlier points
    :return points: list of points
    """

    x = []
    y = []
    z = []

    file_name = path.join(sparse_folder, 'points3D.txt')
    with open(file_name, "r") as f:
        for line in f.readlines():
            if "#" not in line:
                data = line.split(" ")
                x_val = float(data[1])
                y_val = float(data[3])
                z_val = float(data[2])

                # filter outliers
                if abs(x_val) < threshold and abs(y_val) < threshold and abs(z_val) < threshold:
                    x.append(x_val)
                    y.append(y_val)
                    z.append(z_val)

    fig = plt.figure()
    plt.scatter(x, y, linewidth=0.1, s=2)
    fig.savefig(path.join(output_path, 'sparse_plot.png'))


def run_colmap(input_path: str) -> str:
    """
    Create a new sparse model using COLMAP
    """
    path_to_images = path.join(input_path, 'images')

    # run colmap to create model
    subprocess.run(['colmap', 'automatic_reconstructor',
                    '--workspace_path', input_path, '--image_path', path_to_images,
                    '--data_type=video'])

    path_to_sparse_model = path.join(input_path, 'sparse/0')
    plot_model_2d(path_to_sparse_model, input_path)

    path_to_sparse_image = path.join(input_path, 'sparse_plot.png')
    image = Image.open(path_to_sparse_image)
    image.show()

    return path_to_sparse_model


def get_pose_from_file(path_to_folder: str) -> dict:
    """
    Parsing the camera pose from COLMAP image.txt output file
    :param path_to_folder: path to sparse model directory
    :return camera_pose_dict: dictionary of camera pose <camera_id , camera_pose>
    """
    camera_pose_dict = {}
    image_src = path.join(path_to_folder, 'images.txt')

    # read images file
    f = open(image_src, "r")
    lines = f.readlines()[4::2]
    f.close()

    # create absolut camera pose dictionary
    for line in lines:
        columns = line.split()
        image_name = columns[9].split('.')[0]
        image_id = int(image_name.split('e')[1])

        camera_pose_dict[image_id] = []

        # get rotation matrix
        qw = float(columns[1])
        qx = float(columns[2])
        qy = float(columns[3])
        qz = float(columns[4])
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)

        # get translation vector
        tx = float(columns[5])
        ty = float(columns[6])
        tz = float(columns[7])
        T = np.array([tx, ty, tz])

        camera_pose_dict[image_id].append(R)
        camera_pose_dict[image_id].append(T)

    # return camera pose sorted by image id
    return collections.OrderedDict(sorted(camera_pose_dict.items()))


def get_rel_pose(path_to_sparse_model: str, path_to_workspace: str) -> None:
    """
    The function converts absolute camera pose data to relative
    :param path_to_sparse_model: path to sparse model directory
    :param path_to_workspace: path to the main workspace directory
    """
    camera_pose_dict = get_pose_from_file(path_to_sparse_model)
    ordered_ref_data = {}

    prev_rotation, global_rotation = np.identity(3), np.identity(3)
    prev_translation, global_translation = np.zeros(3), np.zeros(3)

    pose_images_path = path.join(path_to_workspace, 'pose_images')
    makedirs(pose_images_path)

    for image in camera_pose_dict.keys():
        # get absolut R & T
        rotation = camera_pose_dict[image][0]
        translation = camera_pose_dict[image][1]

        # compute the relative R & T
        relative_rotation = rotation @ prev_rotation.T
        relative_translation = translation - prev_translation

        # compute global R & T to prevent conversion errors
        global_rotation = global_rotation @ relative_rotation
        global_translation += relative_translation

        ordered_ref_data[image] = []
        ordered_ref_data[image].append(relative_rotation)
        ordered_ref_data[image].append(relative_translation)

        # plot pose images to see if the camera pose are right
        draw_rel_camera_pose(image, translation,
                             (global_rotation + global_translation), pose_images_path)

        prev_rotation = rotation
        prev_translation = translation

    # save camera pose at ORB_Version folder
    dd.io.save('ref_camera_pose.h5', ordered_ref_data)
    webbrowser.open(path.realpath(pose_images_path))


def main(input_path):
    clear_workspace(input_path)

    # extract images from video
    prepare_video(input_path, 87)

    # create model with colmap
    path_to_sparse_model = run_colmap(input_path)

    # ask user for rerun colmap to create new model
    val = input("Do you want to keep this model (T/F)? ")
    if val.lower() == 'f':
        main(input_path)
    else:
        get_rel_pose(path_to_sparse_model, input_path)

        val = input("Do you want to keep this model (T/F)? ")
        if val.lower() == 'f':
            main(input_path)


if __name__ == "__main__":
    # Parse input arguments:
    ap = ArgumentParser(description='Create camera_pose files.')
    ap.add_argument('-p', '--workspace_path', required=True)
    args = vars(ap.parse_args())
    workspace_path = args['workspace_path']

    main(workspace_path)

    print("end camera pose preparation")
