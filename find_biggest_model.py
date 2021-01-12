###################
# 0. General Setup:
###################

import os
import struct
import argparse


# Parser for path to database file:
def parse_args():
    ap = argparse.ArgumentParser(description='Converting colmap output files into ORB descriptors.')
    ap.add_argument('-p', '--input_path', required=True)
    ap.add_argument('-o', '--output_path', type=str, default=None)
    args = vars(ap.parse_args())
    return args['input_path'], args['output_path']


# Read bytes from colmap output file:
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """
    Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def get_points3d_binary(path):
    """
        Get number of 3d points in current input model.
    """
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
    return num_points


########################################################################################################################

##############################################
# 1. Find Model with Most Number of 3D Points:
##############################################

if __name__ == "__main__":

    # Parse input arguments:
    input_path, output_path = parse_args()
    if output_path is None:
        output_path = input_path

    # Get number of points in each output model:
    max_points = get_points3d_binary(os.path.join(input_path, '0/points3D.bin'))
    max_model = 0
    for i in range(1, 10):
        if os.path.isdir(os.path.join(input_path, '{}'.format(i))):
            num_points = get_points3d_binary(os.path.join(input_path, '1/points3D.bin'))
            if num_points > max_points:
                max_model = i
                max_points = num_points
        else:
            break

    # Place model with most points as model - 0:
    if max_model != 0:
        # Rename model-0 with temp:
        os.rename(os.path.join(input_path, '0/'),
                  os.path.join(input_path, 'temp/'))
        # Rename model-max as model-zero:
        os.rename(os.path.join(input_path, '{}/'.format(max_model)),
                  os.path.join(input_path, '0/'))
        # Rename original model-zero as model-max index:
        os.rename(os.path.join(input_path, 'temp/'),
                  os.path.join(input_path, '{}/'.format(max_model)))
