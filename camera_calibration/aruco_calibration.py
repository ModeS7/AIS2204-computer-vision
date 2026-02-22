# https://www.ostirion.net/post/webcam-calibration-with-opencv-directly-from-the-stream
# Import required modules:
import cv2
import numpy as np
from cv2 import aruco
import pickle
import glob
import os
from time import sleep

# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)
RECORD = True

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=0.04,
        markerLength=0.02,
        dictionary=ARUCO_DICT)

# Vector for the 3D points:
threedpoints = []

# Vector for 2D points:
twodpoints = []

# 3D points real world coordinates:
objectp3d = np.zeros((1, CHARUCOBOARD_ROWCOUNT
                      * CHARUCOBOARD_COLCOUNT,
                      3), np.float32)

objectp3d[0, :, :2] = np.mgrid[0:CHARUCOBOARD_ROWCOUNT,
                      0:CHARUCOBOARD_COLCOUNT].T.reshape(-1, 2)
prev_img_shape = None
cap = cv2.VideoCapture(0)
FPS = cap.get(cv2.CAP_PROP_FPS)

corners_all = [] # Corners discovered in all images processed
ids_all = [] # Aruco ids corresponding to corners discovered
image_size = None # Determined at runtime

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

if RECORD:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter('calibration.mp4',
                             cv2.VideoWriter_fourcc(*'DIVX'),
                             FPS,
                             (width, height))

    while True:
        ret, img = cap.read()
        image = img
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find aruco markers in the query image
        corners, ids, _ = aruco.detectMarkers(
            image=gray,
            dictionary=ARUCO_DICT)

        # Outline the aruco markers found in our query image
        img = aruco.drawDetectedMarkers(
            image=img,
            corners=corners)

        # Get charuco corners and ids from detected aruco markers
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=CHARUCO_BOARD)


        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret > 20:
            corners_all.append(charuco_corners)
            ids_all.append(charuco_ids)

            # Refine the pixel coordinates for given 2d points.
            charuco_corners2 = cv2.cornerSubPix(
                image=gray,
                corners=charuco_corners,
                winSize=(3, 3),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS
                          + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001))

            # Draw the refined corner locations
            img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners2,
                charucoIds=charuco_ids)
            # When we have minimum number of data points, stop:
            if len(corners_all) > MIN_POINTS:
                cap.release()
                if RECORD: writer.release()
                cv2.destroyAllWindows()
                break

            # Draw and display the corners:
            image = cv2.drawChessboardCorners(image,
                                              CHECKERBOARD,
                                              corners2, ret)

        cv2.imshow('img', image)

        if RECORD:
            writer.write(image)

        # wait for ESC key to exit and terminate feed.
        k = cv2.waitKey(1)
        if k == 27:
            cap.release()
            if RECORD: writer.release()
            cv2.destroyAllWindows()
            break
    h, w = image.shape[:2]

    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None)

    # Displaying required output
    print(" Camera matrix:")
    print(mtx)

    print("\n Distortion coefficient:")
    print(dist)

    print("\n Rotation Vectors:")
    print(rvecs)

    print("\n Translation Vectors:")
    print(tvecs)

    from numpy import savetxt
    from numpy import genfromtxt

    mean_r = np.mean(np.asarray(rvecs), axis=0)
    mean_t = np.mean(np.asarray(tvecs), axis=0)
    savetxt('rotation_vectors.csv', mean_r, delimiter=',')
    savetxt('translation_vectors.csv', mean_t, delimiter=',')
    savetxt('camera_matrix.csv', mtx, delimiter=',')
    savetxt('camera_distortion.csv', dist, delimiter=',')

    img = cv2.imread('1.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite('calibresult.png', dst)
