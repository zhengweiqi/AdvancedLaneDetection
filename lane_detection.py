#lane detection, main func
import calibration
import utilities
import line
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  
import numpy as np
import hmi

#for output path configuration 
from os.path import basename,exists
from os import makedirs

#for batch files input 
import glob 

#for video input 
from moviepy.editor import VideoFileClip 

#calibrate camera
cameraMatrix, distortionCoeffs = calibration.calibrateCamera()

#
perspectiveMatrix = calibration.perspectiveTransformMatrix()

def getThresholdImg(img):
    #hmi.plotImageWithCoordinateInfo((img, None))
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    combinedBin = np.zeros_like(grayImg)
    combinedBin[(utilities.absSobelThresh(grayImg,'x', (50,255), 5) == 1) | \
                ((utilities.magThresh(grayImg, (50, 255),5)  == 1) & \
                 (utilities.dirThresh(grayImg,(0.7,1.5),15) == 1)) | \
                 (utilities.colorThresh(img,(120,255), (120, 255)) == 1)] = 1
    #blur operation for noise
    combinedBin = cv2.medianBlur(combinedBin, 5)
    #plt.imshow(combinedBin, cmap='gray')
    #applying mask on combined binary image output 
    '''mask = np.zeros_like(combinedBin)
    height=img.shape[0]
    width = img.shape[1]
    regionOfInterest = np.array([[0,height-1], [500,400], [800, 400], [width-1, height-1]], dtype=np.int32)
    cv2.fillPoly(mask, [regionOfInterest], 1)
    thresholdedImg = cv2.bitwise_and(combinedBin, mask)'''
    return combinedBin
    
def imgFindLane(srcImg):
    #undistort source image
    undistortImg = calibration.undistortImage(srcImg, cameraMatrix, distortionCoeffs) 
    
    #combining thresholds to get baseline before perform polynomial fit
    thresholdedImg = getThresholdImg(undistortImg)
    warpedImg = cv2.warpPerspective(thresholdedImg, perspectiveMatrix, thresholdedImg.shape[::-1])
    
    #calculate line's fit, curvature and offset info
    leftFit, rightFit, curvature, offset= line.fitPolynomial(warpedImg)
    
    #get polynomial fit vertices
    plotY = np.linspace(0, undistortImg.shape[0]-1, undistortImg.shape[0]-0)
    leftFitX = leftFit[0]*plotY**2 + leftFit[1]*plotY + leftFit[2]
    rightFitX = rightFit[0]*plotY**2 + rightFit[1]*plotY + rightFit[2]
    
    #apply final drawing on result 
    attachedImg = hmi.finalImgDrawing(srcImg, undistortImg, leftFitX, rightFitX, plotY, perspectiveMatrix, curvature, offset) 
    return attachedImg
    
def videoFindLane(src, dstPath):
    video = VideoFileClip(src)
    videoFindLane = video.fl_image(imgFindLane)
    if dstPath.endswith('.mp4') :
        dirFile = dstPath
    else:
        dirFile = dstPath + '/' + basename(src) 
    videoFindLane.write_videofile(dirFile, audio=False)
 

if __name__ == "__main__":
    
    ''' getting step-by-step output images '''
    #undistortion example
    srcCalibrationImg = mpimg.imread('camera_cal/calibration4.jpg')
    undistortImg = calibration.undistortImage(srcCalibrationImg, cameraMatrix, distortionCoeffs)
    #toggle colorspace before write to image using cv2.imwrite()
    cv2.imwrite('output/cameraCalibration/calibration4Undistort.jpg', cv2.cvtColor(undistortImg, cv2.COLOR_RGB2BGR))
    #hmi.plotImages((srcCalibrationImg,None),(undistortImg, None))
    
    
    #get binary threshold image
    srcImg = mpimg.imread('test_images/test2.jpg')
    undistortImg = calibration.undistortImage(srcImg, cameraMatrix, distortionCoeffs) 
    cv2.imwrite('output/pipelineImages/test2UndistortImg.jpg', cv2.cvtColor(undistortImg, cv2.COLOR_RGB2BGR))
    hmi.plotImages((srcImg,None),(undistortImg, None))
    thresholdedImg = getThresholdImg(undistortImg)
    cv2.imwrite('output/pipelineImages/test2ThresholdImg.jpg', thresholdedImg.astype('uint8')*255)
    #hmi.plotImages((srcImg,None),(thresholdedImg, 'gray'))
    
    #get references for perspectiveMatrix 
    refImg = mpimg.imread('test_images/straight_lines1.jpg')
    #hmi.plotImageWithCoordinateInfo((refImg,None))
    
    #get bird-eye view of thresholded image
    warpedImg = cv2.warpPerspective(thresholdedImg, perspectiveMatrix, thresholdedImg.shape[::-1])
    cv2.imwrite('output/pipelineImages/test2WarpedImg.jpg', warpedImg.astype('uint8')*255)
    #hmi.plotImages((srcImg,None),(warpedImg, 'gray'))
    
    #get drawing of lines after polynomial fit
    lineImg = line.fitPolynomial(warpedImg, True)
    #cv2.imwrite('output/pipelineImages/test2LineImg.jpg', cv2.cvtColor(lineImg, cv2.COLOR_RGB2BGR))
    
    #get projection back to 3d coordinate system 
    leftFit, rightFit, curvature, offset= line.fitPolynomial(warpedImg)
    
    #get polynomial fit vertices
    plotY = np.linspace(0, undistortImg.shape[0]-1, undistortImg.shape[0]-0)
    leftFitX = leftFit[0]*plotY**2 + leftFit[1]*plotY + leftFit[2]
    rightFitX = rightFit[0]*plotY**2 + rightFit[1]*plotY + rightFit[2]
    
    #apply final drawing on result 
    attachedImg = hmi.finalImgDrawing(srcImg, undistortImg, leftFitX, rightFitX, plotY, perspectiveMatrix, curvature, offset)
    cv2.imwrite('output/pipelineImages/test2Final.jpg', cv2.cvtColor(attachedImg, cv2.COLOR_RGB2BGR))
    #hmi.plotImages((srcImg,None),(attachedImg, None))
    
    
    
    '''pipeline of image'''
    srcImg = mpimg.imread('test_images/test4.jpg')  
    retImg = imgFindLane(srcImg)
    cv2.imwrite('output/pipelineImages/test4PipeLine.jpg', cv2.cvtColor(retImg, cv2.COLOR_RGB2BGR))
    #hmi.plotImages((srcImg,None),(retImg, None))
    
    '''pipeline of video'''
    videoFindLane('project_video.mp4', 'output/pipelineVideo/project_video_output.mp4')