###################
# 0. General Setup:
###################

import os
import cv2
import sqlite3
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Parser for path to database file:
def parse_args():
    ap = argparse.ArgumentParser(description='Converting colmap output files into ORB descriptors.')
    ap.add_argument('-p', '--input_path', required=True)
    ap.add_argument('-o', '--output_path', type=str, default=None)
    args = vars(ap.parse_args())
    return args['input_path'], args['output_path']


def get_kp(cur, img_list):
    kp = []
    for img_id, _ in img_list:
        cur.execute('SELECT data FROM keypoints WHERE image_id=?;', (img_id,))
        row = next(cur)
        if row[0] is None:
            kp.append(np.zeros((0, 4), dtype=np.float32))
        else:
            kp_temp = np.fromstring(row[0], dtype=np.float32).reshape(-1, 4)
            kp.append([cv2.KeyPoint(x=k_pt[0], y=k_pt[1], _size=k_pt[2], _angle=k_pt[3]) for k_pt in kp_temp])
    return kp


def get_desc(cur, img_list):
    dsk = []
    for img_id, _ in img_list:
        cur.execute('SELECT data FROM descriptors WHERE image_id=?;', (img_id,))
        row = next(cur)
        if row[0] is None:
            dsk.append(np.zeros((0, 128), dtype=np.uint8))
        else:
            dsk.append(np.fromstring(row[0], dtype=np.uint8).reshape(-1, 128))
    return dsk


def get_3Dpoints(path):
    points = []
    with open(path + 'points3D.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2d_ids = np.array(tuple(map(int, elems[9::2])))
                error = float(elems[7])
                points.append([xyz, image_ids, point2d_ids, error])
    return points


########################################################################################################################

####################################################
# 1. Get keypoints and 3D points from colmap output:
####################################################

if __name__ == "__main__":

    # Parse input arguments:
    input_path, output_path = '/home/rbdstudent/Documents/dani_room/scan2/ORB/', None
    if output_path is None:
        output_path = input_path

    # Connect to the database and create cursor object:
    connection = sqlite3.connect(input_path + 'database.db')
    cursor = connection.cursor()

    # Get list of images (img_id, img_name):
    cursor.execute('SELECT image_id, name FROM images;')
    images = list(row for row in cursor)

    # Get list of keypoints:
    keypoints = get_kp(cursor, images)

    # Get list of descriptors:
    descs = get_desc(cursor, images)

    # End SQL connection:
    cursor.close()
    connection.close()

    # Get list 3D points:
    points_3D = get_3Dpoints(input_path)

    # Compute mean ORB descriptors for all 3D points:
    feature_desc = []
    for p, point in enumerate(points_3D):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        for d in range(min(len(point[1]), 8)):
            img_id = point[1][d]-1
            pt_id = point[2][d]
            img = cv2.imread(os.path.join(input_path, images[img_id][1]))#[..., ::-1]
            if img is None:
                continue
            kp = keypoints[img_id][pt_id]
            x = np.int(kp.pt[0])
            y = np.int(kp.pt[1])
            axs[np.unravel_index(d, (2, 4))].imshow(img)
            axs[np.unravel_index(d, (2, 4))].plot([x], [y], marker='o', markersize=5, color=(0, 1, 0))
            axs[np.unravel_index(d, (2, 4))].set_title('{}'.format(img_id))
        plt.show()
        if p > 20:
            break
