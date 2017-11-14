import numpy as np
import cv2
import glob
import pickle

def save(fname, img):
    output_folder = "./output_images/"
    filename = "{output_folder!s}/{fname!s}".format(**locals())
    cv2.imwrite(filename, img)

def calibrate(image_path='./camera_cal/calibration*.jpg'):
    nx = 9 # the number of inside corners in x
    ny = 6 # the number of inside corners in y
    objpoints = []
    imgpoints = []
    images = glob.glob(image_path)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    chessboard_corners = []

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw the corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            chessboard_corners.append(img)

    save('calibration-corners.jpg', chessboard_corners[12])
    img = cv2.imread('./camera_cal/calibration1.jpg')
    save('calibration.jpg', img)
    img_size = (img.shape[1], img.shape[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    save('calibration-undistorted.jpg', undist)

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open('./calibration_pickle.p', 'wb'))

    return chessboard_corners

chessboard_corners = calibrate()
