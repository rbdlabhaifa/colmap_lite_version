import os
from threading import Thread
from time import sleep
import cv2
from djitellopy import Tello
from enum import Enum
import argparse
import picamera

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--output', required=True,
                help='path to export video')

tello_conn = None
FIFO_Images = "/tmp/images.fifo"


class Triangulation(Enum):
    Up_Down = 0
    Forward_Backward = 1

class Drone(object):
    def __init__(self):
        self.drone = None
        self.drone_speed = 25
        self.connect_drone()

    def __del__(self):
        global tello_conn
        print("tello destroyed")
        self.disconnect_drone()
        tello_conn.close()

    def begin_scan(self, triangulation_type):
        i = 18
        sleep(1)
        while i != 0:
            self.drone.rotate_clockwise(21)
            sleep(2)
            self.do_triangulation(triangulation_type)
            i -= 1
            sleep(3)

    def do_triangulation(self, triangulation_enum):
        if triangulation_enum == Triangulation.Forward_Backward:
            self.drone.move_forward(23)
            sleep(3)
            self.drone.move_back(23)

        if triangulation_enum == Triangulation.Up_Down:
            self.drone.move_up(23)
            sleep(3)
            self.drone.move_down(23)

    def stream_image_to_pipe(self):
        """out = cv2.VideoWriter(os.path.join(output_path, 'outpy.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (960, 720))
        frame_read = self.drone.get_frame_read()
        sleep(2)
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
        out.release()"""
        camera = picamera.PiCamera()
        camera.resolution = (640, 480)
        output_path = vars(ap.parse_args())['output']
        filename = os.path.join(output_path, 'outpy.h264')
        print('Save video to: ', filename)
        camera.start_recording(filename)
        while True:
            camera.wait_recording(1)
        camera.stop_recording()
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
        sleep(3)
        Thread(target=self.stream_image_to_pipe, daemon=True).start()
        self.begin_scan(triangulation_type)
        print("battery after scan:" + self.drone.get_battery())
        self.drone.land()
        sleep(3)


if __name__ == '__main__':
    drone = Drone()
    drone.scan(Triangulation.Up_Down)
    drone.disconnect_drone()
    print("End scan script")
