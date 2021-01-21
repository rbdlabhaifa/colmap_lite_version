from Frame import Frame
from Room import Room


class Point:
    def __init__(self, x, y, z, qx=0, qy=0, qz=0, qw=0, frame_id=0, label=-1):
        """

        :param x: the axis x
        :param y: the axis y
        :param z: the axis z
        :param qx: the quaternion x
        :param qy: the quaternion y
        :param qz: the quaternion z
        :param qw: the quaternion w
        :param frame_id: in order gather all the frame points in a points list
        :param label: in order to find clusters
        """
        self.x = x
        self.y = y
        self.z = z
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw
        self.label = label
        self.frame_id = frame_id

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)


def create_date_from_colmap(images_file_name, point3d_file_name, threshold=1000):
    room = Room()
    with open(images_file_name, "r") as frames_file:
        for i in range(0, 4):
            frames_file.readline()
        row = frames_file.readline()
        even = True
        while row != '':
            if even:
                values = row.split(" ")
                room.frames[int(values[0])] = Frame(float(values[5]), float(values[6]), float(values[7]),
                                                    float(values[2]), float(values[3]), float(values[4]),
                                                    float(values[1]), int(values[0]))
            row = frames_file.readline()
            even = not even
    with open(point3d_file_name, "r") as point3d_file:
        for i in range(0, 3):
            point3d_file.readline()
        row = point3d_file.readline()
        while row != '':
            values = row.split(" ")
            frame = room.frames[int(values[8])]
            x = float(values[1])
            y = float(values[3])
            z = float(values[2])
            if abs(x) < threshold and abs(y) < threshold and abs(z) < threshold:
                room.points.append(Point(x, y, z, frame.qx, frame.qy,
                                         frame.qz, frame.qw, frame.frame_id))
            row = point3d_file.readline()

        print(len(room.points))
    return room.points


def get_x_dimension(points):
    return [point.x for point in points]


def get_y_dimension(points):
    return [point.y for point in points]


def get_z_dimension(points):
    return [point.z for point in points]
