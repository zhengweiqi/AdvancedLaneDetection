import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
#https://mpld3.github.io/examples/mouse_position.html
import mpld3
from mpld3 import plugins
import cv2

#*NOTE* inputParam as tuple, 1st as source image, second as plot type
def plotImageWithCoordinateInfo(imgSrc):
    f, ax = plt.subplots()
    plugins.connect(f, plugins.MousePosition(fontsize=20))
    ax.imshow(imgSrc[0], cmap=imgSrc[1])
    mpld3.show()
    
#*NOTE* inputParam as tuple, 1st as source image, second as plot type
def plotImages(img1Src, img2Src):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    f.tight_layout()
    ax1.imshow(img1Src[0], cmap = img1Src[1])
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(img2Src[0], cmap = img2Src[1])
    ax2.set_title('Target Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    


def finalImgDrawing(srcImg, undistortImg, leftFitX, rightFitX, plotY, persTransMatrix, curvature, offset):
    colorWarp = np.zeros((srcImg.shape[0], srcImg.shape[1], 3), dtype='uint8')    
    ptsLeft = np.array([np.transpose(np.vstack([leftFitX, plotY]))])
    ptsRight = np.array([np.flipud(np.transpose(np.vstack([rightFitX, plotY])))])
    pts = np.hstack((ptsLeft, ptsRight))
    cv2.fillPoly(colorWarp, np.int_([pts]), (0,255, 0))
    
    #project detected lanes onto 3d coordinates system 
    newWarp = cv2.warpPerspective(colorWarp, np.linalg.inv(persTransMatrix), (srcImg.shape[1], srcImg.shape[0])) 
    
    #combine the result with the original image
    result = cv2.addWeighted(undistortImg, 1, newWarp, 0.3, 0)
    
    #add curvature info 
    infoLabelCur = 'Radius Of Curvature: %.2fm' % curvature
    result = cv2.putText(result, infoLabelCur, (50,50),0,1,(255,255,255),2,cv2.LINE_AA)

    #add offset info
    infoLabelOff = 'Vehicle Offset to Lane Center is : %.2fm' % offset   
    result = cv2.putText(result, infoLabelOff, (50,100),0,1,(255,255,255),2,cv2.LINE_AA)
    #plt.imshow(result)
    return result
    