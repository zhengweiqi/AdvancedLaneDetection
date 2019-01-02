#utility functions for debugging and plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#*NOTE* constants definition used for finding lines 
WIN_NUM = 12
WIN_MARGIN = 100
MIN_PIX = 50

X_CORNER = 9
Y_CORNER = 6

#*NOTE* pre-requisite: image has already been converted to gray scale
def absSobelThresh(gray, orient='x', thresh=(0,255), kernelSize=3):
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernelSize) if orient == 'x' else cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    absSobel = np.absolute(sobel)
    scaleSobel = np.uint8(255*absSobel/np.max(absSobel))
    binaryOutput = np.zeros_like(scaleSobel)
    binaryOutput[(scaleSobel >= thresh[0]) & (scaleSobel <= thresh[1])] = 1
    return binaryOutput

#*NOTE* pre-requisite: image has already been converted to gray scale
def magThresh(gray, thresh=(0,255), kernelSize=3):
    #take gradient in x and y separately 
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=kernelSize)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=kernelSize)   
    #calculate magnitude
    sobel = np.sqrt( sobelX**2 + sobelY**2 )
    scaleSobel = np.uint8(255*sobel/np.max(sobel))
    binaryOutput = np.zeros_like(scaleSobel)
    binaryOutput[(scaleSobel >= thresh[0]) & (scaleSobel <= thresh[1])] = 1
    return binaryOutput
  
#*NOTE* pre-requisite: image has already been converted to gray scale
def dirThresh(gray, thresh=(0.0, 0.0), kernelSize=3):
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=kernelSize)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=kernelSize)
    #take the absolute value of the x and y gradients
    absSobelX = np.abs(sobelX)
    absSobelY = np.abs(sobelY)
    #use np.arctan2 to calculate the direction of the gradient 
    direction = np.arctan2(absSobelY,absSobelX)
    binaryOutput = np.zeros_like(direction)
    binaryOutput[(direction>=thresh[0])&(direction<=thresh[1])]=1
    return binaryOutput

#*NOTE* pre-requisite: image shall be in color format RGB
def colorThresh(img, lThresh=(0,255), sThresh=(0,255)):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lVal = hls[:,:,1]
    sVal = hls[:,:,2]
    binaryOutput = np.zeros_like(sVal)
    binaryOutput[ ((lVal>=lThresh[0])&(lVal<=lThresh[1])) & ((sVal>=sThresh[0])&(sVal<=sThresh[1]))]=1
    return binaryOutput
    

