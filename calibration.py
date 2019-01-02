#calibration for images in camera_cal
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import utilities #getting project constant values

# termination criteria for subpix in camera calibration 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrateCamera():
    objpoints = []
    imgpoints = []
    files = glob.glob('camera_cal/calibration*.jpg') #glob src images
    #get shape of images 
    shape = ()
    #prepare object points in advance 
    objp = np.zeros((utilities.X_CORNER*utilities.Y_CORNER,3), np.float32)
    objp[:,:2] = np.mgrid[0:utilities.X_CORNER, 0:utilities.Y_CORNER].T.reshape(-1,2)
    
    #collect image points and object points
    for file in files:
        img = mpimg.imread(file)
        if not shape:
            shape = img.shape[::-1] #get shape of images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray scale 
        ret, corners = cv2.findChessboardCorners(gray, (utilities.X_CORNER,utilities.Y_CORNER), None)    
        if ret == True:
            objpoints.append(objp) #append to objpoints
            cornersops = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) #use subpixeling for accuracy
            imgpoints.append(cornersops) #append to imgpoints 
    ret, cameraMatrix, distortionCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return (cameraMatrix, distortionCoeffs)
        
def undistortImage(img, cameraMatrix, distortionCoeffs):
    undistortImg = cv2.undistort(img, cameraMatrix, distortionCoeffs, None, cameraMatrix)
    return undistortImg
    
def perspectiveTransformMatrix():
    #reference coordinates from straight_lines1.jpg
     src = np.float32([[250,680],[520,500],[760,500],[1040,680]])
     dst = np.float32([[250,680],[250,500],[1040,500],[1040,680]])    
     return cv2.getPerspectiveTransform(src, dst)
 