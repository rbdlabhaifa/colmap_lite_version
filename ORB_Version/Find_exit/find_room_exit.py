# Parser definition:
import argparse
import os
from Point import create_date_from_colmap
from source import get_exit_point

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True,
                help='path to colmap output files')


def get_input_arguments(debug=False):
    if debug:
        return 'images.txt', 'points3D.txt'
    args = vars(ap.parse_args())
    path = args['path']
    images_path = os.path.join(path, 'images.txt')
    points_path = os.path.join(path, 'points3D.txt')

    return path, images_path, points_path


if __name__ == '__main__':
    output_path, images_input_file, points_input_file = get_input_arguments()
    #print('read points from',points_input_file)
    #print('read images from',images_input_file)
    #print('save output to',output_path)
    data = create_date_from_colmap(images_input_file, points_input_file)
    get_exit_point(data, False, True, output_path)
    pass

