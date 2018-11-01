import re
import pickle
import argparse

import cv2
import cv2.aruco as aruco
import numpy

def list_aruco_dicts() :
    prefix = 'DICT_'
    for k,v in cv2.aruco.__dict__.items() :
        if k.startswith(prefix) and isinstance(v,int) :
            yield k[len(prefix):], v


def find_aruco_dict(name):
    for k,v in list_aruco_dicts():
        if k.lower() == name.lower():
            return aruco.Dictionary_get(v)


def board_from_name(name, square_length=25):
    m = re.match(r'(aruco|chArUco)-(\w+)-(\d+)x(\d+)(-(0?\.\d+))?', name, re.I)
    if m:
        dictionary = find_aruco_dict(m.group(2))
        if not dictionary:
            raise ValueError('no dictionary named %s' % m.group(2))
        w = int(m.group(3))
        h = int(m.group(4))
        ratio = float(m.group(6)) if m.group(6) else .5
        if m.group(1).lower() == 'charuco':
            return cv2.aruco.CharucoBoard_create(w, h, square_length, square_length*ratio, dictionary)
        else:
            return cv2.aruco.GridBoard_create(w, h, square_length, square_length*ratio, dictionary)

    raise ValueError('cannot create board %s' % name)


def board_size(board):
    if isinstance(board, cv2.aruco_CharucoBoard):
        return board.getChessboardSize()
    if isinstance(board, cv2.aruco_GridBoard):
        return board.getGridSize()
    raise ValueError('cannot get board size for %s' % board)




def constant_z_unproject(rvec, tvec, camera_matrix, dist_coeffs, z):
    """from https://stackoverflow.com/q/12299870"""
    rotation_matrix, _jac = cv2.Rodrigues(rvec)
    rotation_matrix_inv = numpy.linalg.inv(rotation_matrix)
    camera_matrix_inv = numpy.linalg.inv(camera_matrix)
    rotation_inv_camera_inv = rotation_matrix_inv @ camera_matrix_inv
    right_side = rotation_matrix_inv @ tvec

    def unproject(ij):
        i,j = ij
        uvPoint = numpy.array([[i], [j], [1]])
        left_side  = rotation_inv_camera_inv @ uvPoint
        s = (z + right_side[2][0]) / left_side[2]
        p = rotation_matrix_inv @ (s * camera_matrix_inv @ uvPoint - tvec)
        return p[0][0],p[1][0]

    return unproject



if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    subparser = subparsers.add_parser('print')
    subparser.add_argument('board')

    subparser = subparsers.add_parser('calibrate')
    subparser.add_argument('board')
    subparser.add_argument('images', nargs='+')
    
    subparser = subparsers.add_parser('test')
    subparser.add_argument('board')
    subparser.add_argument('image')

    args = parser.parse_args()

    if args.command == 'print':
        board = board_from_name(args.board)
        w,h = board_size(board)
        img = board.draw((128*w, 128*h))
        fn = '%s.png' % args.board
        cv2.imwrite(fn, img)
        print('saved %s' % fn)



    if args.command == 'test':
        board = board_from_name(args.board)
        _err, camera_matrix, dist_coeffs, _rvecs, _tvecs = pickle.load(open('calibration.pkl','rb'))

        img = cv2.imread(args.image)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, board.dictionary)
        img_aruco = aruco.drawDetectedMarkers(img, corners, ids)

        retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs)
        if retval:
            print('pose ok')
            l = 0.01
            img_aruco = aruco.drawAxis(img_aruco, camera_matrix, dist_coeffs, rvec, tvec, l)
        cv2.imwrite('test.jpg', img_aruco)



    if args.command == 'calibrate':
        board = board_from_name(args.board)
        
        all_corners = []
        all_ids = []
        imsize = None
        for i,fn in enumerate(args.images) :
            img = cv2.imread(fn)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if imsize==None or imsize == gray.shape:
                imsize = gray.shape
                markers = cv2.aruco.detectMarkers(gray, board.dictionary)
                if len(markers[0])>0:
                    markers2 = cv2.aruco.interpolateCornersCharuco(markers[0], markers[1], gray, board)
                    if markers2[1] is not None and markers2[2] is not None and len(markers2[1])>3 :
                        print(fn, 'ok')
                        all_corners.append(markers2[1])
                        all_ids.append(markers2[2])
            else:
               print(fn, 'ignored')

        print('computing calibration...')
        cal = aruco.calibrateCameraCharuco(all_corners, all_ids, board, imsize, None, None)
        #  err, camera_matrix, dist_coeffs, rvecs, tvecs
        fn = 'calibration.pkl'
        print('done (err: %f)' %cal[0])
        pickle.dump(cal, open(fn, 'wb'))

