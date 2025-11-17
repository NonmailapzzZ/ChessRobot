import cv2
import glob
import numpy as np
import os

def calibration() :
    chessboardsize = (9,6)
    framesize = (1280,720)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
    # prepare object point 
    object = np.zeros((chessboardsize[0] * chessboardsize[1], 3), np.float32)
    object[:,:2] = np.mgrid[0:chessboardsize[0], 0:chessboardsize[1]].T.reshape(-1,2)

    # Arrays to store object and image points from all the image
    objectPoint = [] # 3d real world space
    imgPoint = [] # 2d point in image plane
        
    image = r'C:\Users\ASUS\Desktop\ChessRobot\calibration'
    for image in os.listdir(image) :
        print(image)
        img = cv2.imread(os.path.join(r'C:\Users\ASUS\Desktop\ChessRobot\calibration',image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find chess board corner
        ret, corners = cv2.findChessboardCorners(gray, chessboardsize, None)

        # If found, add object point, image point (after refining item)
        if ret == True :
            objectPoint.append(object)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgPoint.append(corners)

            # Draw and display the corners 
            cv2.drawChessboardCorners(img, chessboardsize, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1000)
                
    cv2.destroyAllWindows()
    
    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoint, imgPoint, framesize, None, None)

    img = cv2.imread(r"C:\Users\ASUS\Desktop\ChessRobot\calibration\img_007.jpg")

calibration()