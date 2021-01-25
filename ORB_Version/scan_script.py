import os
from threading import Thread
from time import sleep
import cv2
from djitellopy import Tello
from enum import Enum
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--output', required=True,
                help='path to export video')


def get_input_arguments(debug=False):
    if debug:
        return '/'
    args = vars(ap.parse_args())
    path = args['output']

    return path

tello_conn = None
rotate_clockwise = True
FIFO_Images = "/tmp/images.fifo"


class Triangulation(Enum):
    Up_Down = 0
    Forward_Backward = 1

class Drone(object):
    def __init__(self):
        self.drone = None
        self.drone_speed = 25
        self.connect_drone()
        battery = self.drone.get_battery()
        print("before scan:" + str(battery))
        if battery > 30:
            Thread(target=self.stream_image_to_pipe, daemon=True).start()
        else:
            print("not enough battery")

    def __del__(self):
        global tello_conn
        print("tello destroyed")
        self.disconnect_drone()
        tello_conn.close()

    def begin_scan(self, triangulation_type):
        i = 20
        self.drone.move_down(25)
        sleep(1)
        while i != 0:
            self.drone.rotate_clockwise(18)
            self.do_triangulation(triangulation_type)
            i -= 1
            sleep(1)
        self.drone.send_rc_control(0, 0, 0, 0)

    def do_triangulation(self, triangulation_enum):
        if triangulation_enum == Triangulation.Forward_Backward:
            self.drone.move_forward(20)
            sleep(1)
            self.drone.move_back(20)

        if triangulation_enum == Triangulation.Up_Down:
            self.drone.move_up(20)
            sleep(1)
            self.drone.move_down(23)

        sleep(1)

    def stream_image_to_pipe(self):
        output_path = get_input_arguments()
        out = cv2.VideoWriter(os.path.join(output_path, 'outpy.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (960, 720))
        frame_read = self.drone.get_frame_read()
        try:
            os.mkfifo(FIFO_Images)
        except Exception as e:
            print(e.args[0])
        while True:
            sleep(0.05)
            try:
                current_frame = frame_read.frame
                out.write(current_frame)
            except Exception as e:
                print(e.args[0])
                continue
        out.release()
        print("Created video file")

    def connect_drone(self):
        self.drone = Tello()
        if not self.drone.connect():
            raise RuntimeError("drone not connected")
        self.drone.streamon()

    def disconnect_drone(self):
        self.drone.streamoff()
        self.drone = None

    def scan(self, triangulation_type):
        self.drone.takeoff()
        sleep(1)
        self.begin_scan(triangulation_type)
        battery_level = self.drone.get_battery()
        print("battery after scan:" + str(battery_level))
        self.drone.land()
        sleep(5)
        return


if __name__ == '__main__':
    drone = Drone()
    drone.scan(Triangulation.Up_Down)
    drone.disconnect_drone()
    print("End scan script")

