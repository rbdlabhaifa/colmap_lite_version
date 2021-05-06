import os
import cv2
import sqlite3
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Parser for path to database file:
def parse_args() -> tuple:
    """
    Function to parse user argument
    :return: input_path and output_path
    """
    ap = argparse.ArgumentParser(description='Converting colmap output files into ORB descriptors.')
    ap.add_argument('-p', '--input_path', required=True)
    ap.add_argument('-o', '--output_path', type=str, default=None)
    args = vars(ap.parse_args())
    return args['input_path'], args['output_path']


def get_descriptors(cursor, img_list: list) -> list:
    """
    get list of descriptors for each image
    :param cursor: sqlite3.connect().cursor
    :param img_list: list of images
    :return dsk: list of descriptors
    """
    dsk = []
    for image_id, _ in img_list:
        cursor.execute('SELECT data FROM descriptors WHERE image_id=?;', (image_id,))
        row = next(cursor)
        if row[0] is None:
            dsk.append(np.zeros((0, 128), dtype=np.uint8))
        else:
            dsk.append(np.frombuffer(row[0], dtype=np.uint8).reshape(-1, 128))
    return dsk


def get_3d_points(input_path: str) -> list:
    """
    get list of 3d points form points.txt file
    :param input_path: path to points.txt file
    :return points: list of points
    """
    points = []
    path = os.path.join(input_path, 'points3D.txt')
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                xyz = np.array(tuple(map(float, elements[1:4])))
                image_ids = np.array(tuple(map(int, elements[8::2])))
                point2d_ids = np.array(tuple(map(int, elements[9::2])))
                error = float(elements[7])
                points.append([xyz, image_ids, point2d_ids, error])
    return points


def plot_2d(file_name: str, output_path: str, threshold=500) -> None:
    """
    get list of 3d points form points.txt file
    :param file_name: path to points.txt file
    :param output_path: path to save the plot
    :param threshold: threshold for filter outlier points
    :return points: list of points
    """
    x = []
    y = []
    z = []

    with open(file_name, "r") as f:
        for line in f.readlines():
            if "#" not in line:
                data = line.split(" ")
                x_val = float(data[0])
                y_val = float(data[2])
                z_val = float(data[1])

                # filter outliers
                if abs(x_val) < threshold and abs(y_val) < threshold and abs(z_val) < threshold:
                    x.append(x_val)
                    y.append(y_val)
                    z.append(z_val)

    fig = plt.figure()
    plt.scatter(x, y, linewidth=0.1, s=2)
    fig.savefig(os.path.join(output_path, 'sparse_plot.png'))


def main():
    plot_map = True

    # Parse input arguments:
    input_path, output_path = parse_args()
    if output_path is None:
        output_path = input_path

    # Connect to the database and create cursor object:
    connection = sqlite3.connect(os.path.join(input_path, 'database.db'))
    cursor = connection.cursor()

    # Get list of images (img_id, img_name):
    cursor.execute('SELECT image_id, name FROM images;')
    images = list(row for row in cursor)

    # Get list of descriptors:
    descriptors = get_descriptors(cursor, images)

    # End SQL connection:
    cursor.close()
    connection.close()

    # Get list 3D points:
    points_3D = get_3d_points(input_path)

    # Compute mean ORB descriptors for all 3D points:
    feature_desc = []

    for point in points_3D:
        dsc = []
        for d in range(len(point[1])):
            img_id = point[1][d] - 1
            pt_id = point[2][d]
            if pt_id < len(descriptors[img_id]):
                dsc.append(descriptors[img_id][pt_id])

        if len(dsc) != 0:
            mean_dsc = np.mean(dsc, axis=0)
            l2_dist = np.linalg.norm(mean_dsc - dsc, axis=1)
            feature_desc.append(np.concatenate([point[0], dsc[np.argmin(l2_dist)]]))

    # Save according to RBD requirement:
    feature_desc = np.array(feature_desc)

    np.savetxt(os.path.join(output_path, 'pointData.csv'),
               feature_desc[:, :3], delimiter=',')

    if plot_map:
        np.savetxt(os.path.join(output_path, 'sparse.xyz'),
                   feature_desc[:, :3], delimiter=' ')

        plot_2d(os.path.join(output_path, 'sparse.xyz'), output_path)

    cv_file = cv2.FileStorage(os.path.join(output_path, 'descriptorsData.xml'),
                              cv2.FILE_STORAGE_WRITE)

    for p in range(len(feature_desc)):
        cv_file.write('desc{}'.format(p + 1), feature_desc[p:p + 1, 3:35].astype('uint8'))

    cv_file.release()

    print('Files were saved at ', output_path)


####################################################
# 1. Get keypoints and 3D points from colmap output:
####################################################
if __name__ == "__main__":
    main()
