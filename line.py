import numpy as np
import utilities
import matplotlib.pyplot as plt
import collections

#define window class to hold x, y, width, height position 
#only 2 points are des
class Window:
    def __init__(self):
        self.left_top = None
        self.right_bottom = None
        
        
#define a class to receive the characteristics of each line detection
#*Note: ring buffer mechanism has been used, in order to smoothing output within 5 frames *
class Line():
    def __init__(self):
        #x value for base position
        self.line_base = 0
        #selected indice info 
        self.indices = None
        #x and y values 
        self.all_x = []
        self.all_y = []
        #line fit info
        self.fit = collections.deque(maxlen=5)
        self.fit_avg = []
        #curvature info
        self.radius_of_curvature = None
        #offset info
        self.offset = None
        
    def addFit(self, newFit):
        #adding new fit result to ring buffer, update average fitting result 
        self.fit.append(newFit)
        self.fit_avg = np.average(np.asarray(self.fit), axis=0)
 
#global singleton line objects       
leftLine = Line()
rightLine = Line() 

#*NOTE*: lines global singleton implementation  
def findLanePixels(binWarped):
    global leftLine
    global rightLine
    #take a histogram of bottom half of the image
    hist = np.sum(binWarped[binWarped.shape[0]//2:,:], axis=0)
    outImg = np.dstack((binWarped,binWarped,binWarped))
    #find middle point of histogram, in order to separate to left and right lines
    mid = np.int(hist.shape[0]//2)
    leftLine.line_base = np.argmax(hist[:mid])
    rightLine.line_base = np.argmax(hist[mid:]) + mid
    #set height for windows
    winHeight = np.int(binWarped.shape[0]//utilities.WIN_NUM) 
    nonZero = binWarped.nonzero()
    nonZeroY = np.array(nonZero[0])
    nonZeroX = np.array(nonZero[1])
    #print(np.max(nonZeroX))
    leftLineCur = leftLine.line_base
    rightLineCur = rightLine.line_base
    
    leftWin = Window()
    rightWin = Window()
    
    leftLine.indices = []
    rightLine.indices = []
    
    #step through windows one by one 
    for winIdx in range(utilities.WIN_NUM):
        leftWin.left_top = ((leftLineCur - utilities.WIN_MARGIN), binWarped.shape[0]-(winIdx+1)*winHeight)
        leftWin.right_bottom = ((leftLineCur + utilities.WIN_MARGIN), binWarped.shape[0]-winIdx*winHeight)
        
        rightWin.left_top = ((rightLineCur - utilities.WIN_MARGIN),binWarped.shape[0]-(winIdx+1)*winHeight)
        rightWin.right_bottom =  ((rightLineCur + utilities.WIN_MARGIN),binWarped.shape[0]-winIdx*winHeight)
        #identify valid (non-zero) pixels within left and right windows
        validPixLeft = ((nonZeroY >= leftWin.left_top[1]) & (nonZeroY < leftWin.right_bottom[1]) & \
                        (nonZeroX >= leftWin.left_top[0]) &  (nonZeroX < leftWin.right_bottom[0])).nonzero()[0]
        
        validPixRight = ((nonZeroY >= rightWin.left_top[1]) & (nonZeroY < rightWin.right_bottom[1]) & \
                         (nonZeroX >= rightWin.left_top[0]) &  (nonZeroX < rightWin.right_bottom[0])).nonzero()[0]

        leftLine.indices.append(validPixLeft)
        rightLine.indices.append(validPixRight)
        
        if len(validPixLeft) > utilities.MIN_PIX :
            leftLineCur = np.int(np.mean(nonZeroX[validPixLeft]))
        if len(validPixRight) > utilities.MIN_PIX :
            rightLineCur = np.int(np.mean(nonZeroX[validPixRight]))
    
    # concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        leftLine.indices = np.concatenate(leftLine.indices)
        rightLine.indices = np.concatenate(rightLine.indices)
    except ValueError:
        # avoids an error if the above is not implemented fully
        pass
    
    leftLine.all_x = nonZeroX[leftLine.indices]
    leftLine.all_y = nonZeroY[leftLine.indices]
    rightLine.all_x = nonZeroX[rightLine.indices]
    rightLine.all_y = nonZeroY[rightLine.indices]
    return outImg

#*NOTE*: lines global singleton implementation       
def findLanePixelsAround(binWarped):
    global leftLine
    global rightLine
    nonZero = binWarped.nonzero()
    nonZeroY = np.array(nonZero[0])
    nonZeroX = np.array(nonZero[1])
    outImg = np.dstack((binWarped,binWarped,binWarped))
    leftLine.indices =[]
    rightLine.indices = []

    validPixLeft = ((nonZeroX > (leftLine.fit_avg[0]*(nonZeroY**2) + leftLine.fit_avg[1]*nonZeroY + leftLine.fit_avg[2] - utilities.WIN_MARGIN)) \
                        & (nonZeroX < (leftLine.fit_avg[0]*(nonZeroY**2) + leftLine.fit_avg[1]*nonZeroY + leftLine.fit_avg[2] + utilities.WIN_MARGIN)))
        
    validPixRight = ((nonZeroX > (rightLine.fit_avg[0]*(nonZeroY**2) + rightLine.fit_avg[1]*nonZeroY + rightLine.fit_avg[2] - utilities.WIN_MARGIN)) \
                        & (nonZeroX < (rightLine.fit_avg[0]*(nonZeroY**2) + rightLine.fit_avg[1]*nonZeroY + rightLine.fit_avg[2] + utilities.WIN_MARGIN)))
        
    leftLine.indices.append(validPixLeft)
    rightLine.indices.append(validPixRight)
    leftLine.all_x = nonZeroX[leftLine.indices]
    leftLine.all_y = nonZeroY[leftLine.indices]
    rightLine.all_x = nonZeroX[rightLine.indices]
    rightLine.all_y = nonZeroY[rightLine.indices]
    
    
    return outImg

            
#*NOTE*: lines global singleton implementation 
def fitPolynomial(binWarped, plotLine=False):
    global leftLine
    global rightLine
    if not leftLine.fit :
        findLanePixels(binWarped)
    else :
        findLanePixelsAround(binWarped)

    leftLine.addFit((np.polyfit(leftLine.all_y, leftLine.all_x, 2)))
    rightLine.addFit((np.polyfit(rightLine.all_y, rightLine.all_x, 2)))

    curvature = measureCurvature(leftLine, rightLine)
    offset = measureOffset(leftLine, rightLine)
    if plotLine == True:
        outImg = np.dstack((binWarped, binWarped, binWarped))
        #generate x and y values for plotting
        plotY = np.linspace(0, binWarped.shape[0]-1, binWarped.shape[0])
        
        try:
            leftFitX = leftLine.fit_avg[0]*plotY**2 + leftLine.fit_avg[1]*plotY + leftLine.fit_avg[2]
            rightFitX = rightLine.fit_avg[0]*plotY**2 + rightLine.fit_avg[1]*plotY + rightLine.fit_avg[2]
        except TypeError:
            # avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            leftFitX = 1*plotY**2 + 1*plotY
            rightFitX = 1*plotY**2 + 1*plotY
            
        outImg[leftLine.all_y, leftLine.all_x] = [255, 0, 0]
        outImg[rightLine.all_y, rightLine.all_x] = [0, 0, 255]
    
        # Plots the left and right polynomials on the lane lines
        #plt.plot(leftFitX, plotY, color='yellow')
        #plt.plot(rightFitX, plotY, color='yellow')
        #plt.imshow(outImg)
        return outImg
    else:
        return leftLine.fit_avg, rightLine.fit_avg, curvature, offset
    
def measureCurvature(leftLine, rightLine):
    # Define conversions in x and y from pixels space to meters
    ymPerPix = 30/720 # meters per pixel in y dimension
    xmPerPix = 3.7/700 # meters per pixel in x dimension
    
    maxY = 719 # as it's determined to be 1280*720 by shape 
    
    leftFitReal = np.polyfit(leftLine.all_y*ymPerPix, leftLine.all_x*xmPerPix, 2)
    rightFitReal = np.polyfit(rightLine.all_y*ymPerPix, rightLine.all_x*xmPerPix, 2)

    leftCur = pow((1+(2*leftFitReal[0]*maxY*ymPerPix+leftFitReal[1])**2),1.5) / abs(2*leftFitReal[0])
    rightCur = pow((1+(2*rightFitReal[0]*maxY*ymPerPix+rightFitReal[1])**2),1.5) / abs(2*rightFitReal[0])
    
    return np.mean([leftCur, rightCur])
    
def measureOffset(leftLine, rightLine):
    maxY = 719 # as it's determined to be 1280*720 by shape
    width = 1280
    xmPerPix = 3.7/700 # meters per pixel in x dimension
    
    leftXBottom = leftLine.fit_avg[0]*(maxY**2) + leftLine.fit_avg[1]*maxY + leftLine.fit_avg[2]
    rightXBottom = rightLine.fit_avg[0]*(maxY**2) + rightLine.fit_avg[1]*maxY + rightLine.fit_avg[2]
    
    offsetByPix = width/2 - (leftXBottom + rightXBottom )/2
    offset = offsetByPix * xmPerPix
    return offset
    
    