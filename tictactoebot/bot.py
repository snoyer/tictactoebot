import time
from collections import namedtuple
import pickle
import urllib.request
import math
import itertools
import threading
import logging
import ast
import argparse

import requests
import numpy
import numpy as np
import cv2
import cv2.aruco as aruco


from .vision import analyze_games, find_games, draw_polylines
from . import arucoutil




def main(args):
    aruco_board = arucoutil.board_from_name('charuco-4x4_50-20x2', 25)
    calibration = pickle.load(open(args.calibration,'rb'))
    ox,oy = args.pen_offset

    paper_size = 309, 218

    w,h = paper_size
    l = 10
    paper_corners = [
        [(0,l), (0,0), (l,0)],
        [(w-l,0), (w,0), (w,l)],
        [(w,h-l), (w,h), (w-l,h)],
        [(l,h), (0,h), (0,h-l)],
    ]

    drawing_lock = threading.Lock()




    def process_frame(frame, play=True, img_out=None, lw=1):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        projections = compute_projections(frame_gray, aruco_board, calibration, args.pen_offset)

        if projections:
            old_drawings, new_drawings = analyze_games(find_games(frame_gray), play=play)

            if not img_out is None:
                o = projections.world_to_image(( 0, 0, 0))
                x = projections.world_to_image((25, 0, 0))
                y = projections.world_to_image(( 0,25, 0))
                draw_polylines(img_out, [[o,x]], (0,255,255), lw, cv2.LINE_AA, arrows=True)
                draw_polylines(img_out, [[o,y]], (0,255,255), lw, cv2.LINE_AA, arrows=True)

                w,h = paper_size
                draw_polylines(img_out, [map(projections.paper_to_image, [(0,0),(w,0),(w,h),(0,h),(0,0)])], (0,255,255), 1, cv2.LINE_AA)
                draw_polylines(img_out, mapmap(projections.paper_to_image, paper_corners), (0,255,255), lw*2, cv2.LINE_AA)

                draw_polylines(img_out, old_drawings, (255,0,0), lw*2, cv2.LINE_AA)
                draw_polylines(img_out, new_drawings, (0,0,255), lw*3, cv2.LINE_AA)

            if args.draw_url and new_drawings:
                if drawing_lock.acquire(blocking=False):

                    def draw():
                        try:
                            lines = [[projections.image_to_paper((ij.real,ij.imag)) for ij in ijs] for ijs in new_drawings]
                            w,h = paper_size
                            if all(all(0<=x<=w and 0<=y<=h for x,y in xys) for xys in lines):
                                requests.post(args.draw_url, json=lines)
                            else:
                                logging.info('drawing target is out of bounds')
                        finally:
                            drawing_lock.release()
                    t = threading.Thread(target=draw)
                    t.start()
                else:
                    logging.debug('already drawing')

    if args.mjpeg_url:
        cv2.namedWindow('capture', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('capture', 1280,720)
        cv2.namedWindow('motion')

        cap = MjpegCapture(args.mjpeg_url)
        motiondetect = MotionDetector(alpha=.2, scale=.1)

        while True:
            frame = cap.frame
            if frame is None:
                time.sleep(.1)
                continue

            motiondetect.new_frame(frame)

            try:
                img_out = frame.copy()
                process_frame(frame, motiondetect.is_quiet, img_out)

                cv2.imshow('capture', img_out)
                cv2.imshow('motion', motiondetect.delta)


                key = cv2.waitKey(1)
                if key == -1:
                    pass
                elif key == ord('s'):
                    fn = 'frame-%d.jpg' % time.time()
                    cv2.imwrite(fn, img_out)
                    logging.info('saved %s', fn)
                elif key == 27:
                    logging.debug('break')
                    break

            except cv2.error as e:
                logging.error('opencv error', exc_info=e)

        logging.debug('stopping')
        cap.stop()

    elif args.image_path:
        frame = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
        img_out = frame.copy()
        process_frame(frame, True, img_out, 2)

        fn = 'frame-%d.jpg' % time.time()
        cv2.imwrite(fn, img_out)
        logging.info('saved %s', fn)






Projections = namedtuple('Projections', 'world_to_image image_to_worldplane paper_to_image image_to_paper')

def compute_projections(img_gray, aruco_board, calibration, pen_offset):
    _err, camera_matrix, dist_coeffs, _rvecs, _tvecs = calibration
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_board.dictionary)
    retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, aruco_board, camera_matrix, dist_coeffs)

    if retval:
        def world_to_image(xyz):
            pts = numpy.array([xyz], dtype=numpy.float32)
            imgpts, jac = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeffs)
            return tuple(imgpts[0][0])

        image_to_worldplane = arucoutil.constant_z_unproject(rvec, tvec, camera_matrix, dist_coeffs, 0)

        def worldplane_to_paper(xy):
            x,y = xy
            ox,oy = pen_offset
            return x-ox, -y-oy

        def paper_to_worldplane(xy):
            x,y = xy
            ox,oy = pen_offset
            return x+ox, -(y+oy)

        def image_to_paper(ij):
            xy = image_to_worldplane(ij)
            return worldplane_to_paper(xy)

        def paper_to_image(xy):
            x,y = paper_to_worldplane(xy)
            return world_to_image((x,y,0))

        return Projections(world_to_image, image_to_worldplane, paper_to_image, image_to_paper)

    return None





class MjpegCapture(object):
    def __init__(self, url):
        self.frame = None
        self.stop_event = threading.Event()
        t = threading.Thread(target=self._capture, args=(url,))
        t.start()

    def _capture(self, url):
        for img in self.read_mjpeg_stream(url):
            self.frame = img
            if self.stop_event.is_set():
                break

    def stop(self):
        self.stop_event.set()

    @staticmethod
    def read_mjpeg_stream(url):
        stream = urllib.request.urlopen(url)
        stream_bytes = b''
        while True:
            stream_bytes += stream.read(1024)
            a = stream_bytes.find(b'\xff\xd8')
            b = stream_bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                try:
                    jpg = stream_bytes[a:b+2]
                    stream_bytes = stream_bytes[b+2:]
                    yield cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                except cv2.error:
                    pass



class MotionDetector(object):
    def __init__(self, alpha=.1, scale=.5, blur=5):
        self.alpha = alpha
        self.scale = scale
        self.blur = blur

        self.avg = None
        self.is_quiet = True

        self.delta = None

    def new_frame(self, frame):
        w = int(round(frame.shape[1]*self.scale))
        h = int(round(frame.shape[0]*self.scale))
        frame = cv2.resize(frame, (w,h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (self.blur, self.blur), 0)

        if self.avg is None:
            self.avg = np.float32(frame)
        else:
            cv2.accumulateWeighted(frame, self.avg, self.alpha)

        ref_frame = cv2.convertScaleAbs(self.avg)
        delta = cv2.absdiff(ref_frame, frame)
        delta = cv2.threshold(delta, 15, 255, cv2.THRESH_BINARY)[1]

        self.is_quiet = numpy.max(delta)==0

        self.delta =  delta


def mapmap(f, xss):
    return map(lambda xs:map(f,xs), xss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument('--mjpeg-url')
    src_group.add_argument('--image-path')
    parser.add_argument('--draw-url')
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--pen-offset', type=ast.literal_eval, default=(0,0))

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    main(args)
