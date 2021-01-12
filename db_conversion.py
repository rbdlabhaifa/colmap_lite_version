import os
import cv2
import sqlite3
import argparse
import numpy as np

# Parser for path to database file:
def parse_args():
    ap = argparse.ArgumentParser(description='Converting colmap output files into ORB descriptors.')
    ap.add_argument('-p', '--input_path', required=True)
    ap.add_argument('-o', '--output_path', type=str, default=None)
    args = vars(ap.parse_args())
    return args['input_path'], args['output_path']


"""def get_kp(cur, img_list):
    kp = []
    for img_id, _ in img_list:
        cur.execute('SELECT data FROM keypoints WHERE image_id=?;', (img_id,))
        row = next(cur)
        if row[0] is None:
            kp.append(np.zeros((0, 4), dtype=np.float32))
        else:
            kp_temp = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 4)
            kp.append([cv2.KeyPoint(x=k_pt[0], y=k_pt[1], _size=k_pt[2], _angle=k_pt[3]) for k_pt in kp_temp])
    return kp"""


def get_desc(cur, img_list):
    dsk = []
    for img_id, _ in img_list:
        cur.execute('SELECT data FROM descriptors WHERE image_id=?;', (img_id,))
        row = next(cur)
        if row[0] is None:
            dsk.append(np.zeros((0, 128), dtype=np.uint8))
        else:
            dsk.append(np.frombuffer(row[0], dtype=np.uint8).reshape(-1, 128))
    return dsk


def get_3Dpoints():
    points = []
    path = os.path.join(input_path, 'points3D.txt')
    with open(path, 'r') as file:

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


####################################################
# 1. Get keypoints and 3D points from colmap output:
####################################################
if __name__ == "__main__":

    # Parse input arguments:
    input_path, output_path = parse_args()
    if output_path is None:
        output_path = input_path

    # Connect to the database and create cursor object:
    print(input_path)
    print(output_path)
    connection = sqlite3.connect(os.path.join(input_path, 'database.db'))
    cursor = connection.cursor()

    # Get list of images (img_id, img_name):
    cursor.execute('SELECT image_id, name FROM images;')
    images = list(row for row in cursor)

    # Get list of keypoints:
    #keypoints = get_kp(cursor, images)

    # Get list of descriptors:
    descs = get_desc(cursor, images)

    # End SQL connection:
    cursor.close()
    connection.close()

    # Get list 3D points:
    points_3D = get_3Dpoints()

    # Compute mean ORB descriptors for all 3D points:
    feature_desc = []
    counter = 0
    for point in points_3D:
        dsc = []
        for d in range(len(point[1])):
            img_id = point[1][d]-1
            pt_id = point[2][d]
            if pt_id < len(descs[img_id]):
                dsc.append(descs[img_id][pt_id])

        if len(dsc) != 0:
            mean_dsc = np.mean(dsc, axis=0)
            l2_dist = np.linalg.norm(mean_dsc - dsc, axis=1)
            feature_desc.append(np.concatenate([point[0], dsc[np.argmin(l2_dist)]]))

    # Save according to RBD requirement:
    feature_desc = np.array(feature_desc)

    np.savetxt(os.path.join(output_path, 'points.csv'),
               feature_desc[:, :3], delimiter=',')

    np.savetxt(os.path.join(output_path, 'sparse.xyz'),
               feature_desc[:, :3], delimiter=' ')

    cv_file = cv2.FileStorage(os.path.join(output_path, 'descriptors.xml'),
                              cv2.FILE_STORAGE_WRITE)

    for p in range(len(feature_desc)):
        cv_file.write('desc{}'.format(p + 1), feature_desc[p:p + 1, 3:35].astype('uint8'))

    cv_file.release()

    with open(os.path.join(input_path, 'sfm_log.txt'), 'a') as file:
        file.write('Number of 3D points:      {}\n'.format(len(feature_desc)))

    print('Files were saved at "' + output_path + '"')