## Advanced Lane Detection

*This is the demonstration project done for Udacity Nano Degree Program "Self-Driving Car Engineer"*

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[imageCal]: ./output/cameraCalibration/plotCompare.png "Calibration"
[imageSrc]: ./output/pipelineImages/test2.jpg "Source"
[imageUndist]: ./output/pipelineImages//plotCompareUndistort.png "Undistort"
[imageThr]: ./output/pipelineImages/plotCompareThreshold.png "Threshold"
[imageWarp]: ./output/pipelineImages/plotCompareWarped.png "Wapred"
[imageLine]: ./output/pipelineImages/test2LineImg.png "Line"
[imageRef]: ./output/pipelineImages/referencePerspectiveMatrix.png  "Reference"
[imageCur]: ./output/pipelineImages/test2Final.jpg "Final"
[imageFinal]: ./output/pipelineImages/plotCompareFinal.png "Final"
[video]: ./pipelineVideo/project_video_output.mp4 "Video"



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Dependencies
* Python 3.6.4
* Numpy
* OpenCV
* Matplotlib
* mpld3
* glob
* collections

### How-To
The main entrance is `lane_detection.py`, the main function would get step-by-step example images for output. Including camera calibration, image undistortion, perspective transformation, lines polynomial fitting, and finalize drawing. Detailed explanations are listed in the comments area as :
* getting step-by-step output images
* pipeline of image
* pipeline of video

All the related output are saved into [PROJ]/output folder, separated into `cameraCalibration` `pipelineImages` and `pipelineVideo` subfolders.



### Camera Calibration

The code for this step is contained in `calibration.py`

Calibration procedure starts with preparing __object points__, which will be the (x, y, z) coordinates of the chessboard corners in the world. By making use of prepared chessboard calibration images in `camera_cal` folder. First conversion from RGB to grayscale in color space has been done in every calibration image. Then OpenCV's `cv2.findChessboardCorners()` has taken into place in-order to retrieve and collect __image points__. In order to perform a more accurate result, `cv2.cornerSubPix()` has been used to optimize the  __image points__ detection.

After the  __object points__ and  __image points__ identification, ` cv2.calibrateCamera()` has been used to get __distortion matrix__ and __distortion coefficients__

In order to undistort the image captured by this specific camera, `calibration.undistortImage()` has been provided, which accepting the `source image` `distortion matrix` and `distortion coefficients` whenever it's needed.

Here is an example for calibration on `camera_cal/calibration4.jpg`

![alt text][imageCal]
+ *original image* as camera_cal/calibration4.jpg.
+ *target image* as same image after undistortion.

### Pipeline (single images)

To demonstrate this step, example will be done on image sourcing from `test_images/test2.jpg`

![alt text][imageSrc]

##### 1. Undistort image

Making use of calculated distortion matrix and coefficients ( done by functions in `calibration.py`), use `calibration.undistortImage()` to undistort the source image. Output as

![alt text][imageUndist]
+ *original image* as test_images/test2.jpg.
+ *target image* as same image after undistortion.


#### 2. Use combined threshold to get binary image
Use the undistort image as the input, applying multiple filters to get the edge/line detection info(all individual filter functions are done by functions in `utilities.py`, combined logic has been implemented in `lane_detection.getThresholdImg()`). The filters and masks are listed as below:
* Filters
  * Absolute horizontal Sobel operator on the image
  * Sobel operator in both horizontal and vertical directions and calculate its magnitude
  * Sobel operator to calculate the direction of the gradient
  * Convert the image from RGB space to HLS space, and threshold the S and L channel
* Masks:
  * Since the interest area on the road would be more like a "trapezoid", a mask has been apply only to focus the following manipulation on selected area. Output as
* Smooth Filter:
  * `cv2.medianBlur()` has been used in order to smooth the final binary output, to get rid of some sharp noise.

![alt text][imageThr]
+ *original image* as test_images/test2.jpg.
+ *target image* as same image after binary thresholding.  

#### 3. Perspective transformation
In order to get a "bird's-eye view" of the lane, after getting the thresholded image, a perspective transform has been done.

To get the reference __source points__ and __destination points__, "straightline" images in "test_images" folder are been used. To get the coordinates info of the pixels, therefore `hmi.plotImageWithCoordinateInfo()` has been implemented. By making use of mpld3 plugins, X and Y values are been extracted like below example.
![alt text][imageRef]
In 'test_images/straight_lines1.jpg' the blow 4 points are selected as reference points(hardcoded in `calibration.perspectiveTransformMatrix()`):

| Source        | Destination   |
|:-------------:|:-------------:|
| 250, 680      | 250, 680      |
| 520, 500      | 250, 500      |
| 760, 500      | 1040, 500     |
| 1040, 680     | 1040, 680     |

Then by applying ` cv2.getPerspectiveTransform()`, perspective transform matrix has been calculated.

With the perspective transform matrix, an warped image could be transformed in to "bird's-eye view" like below:
![alt text][imageWarp]
+ *original image* as test_images/test2.jpg.
+ *target image* as same image after applying perspective transformation.  

#### 4. Line fitting
__mainly in line.py__

After getting the perspective transformed source image, now a 2nd order polynomial to both left and right lane lines are done. In detailed steps as :
1. set the number of sliding windows of 9 horizontally, and margin as 100 as left/right range of sliding window. 50 as the thresh to determine the validation for window center.
2. histogram calculation for second half of the binary warped image, and start doing a bottom-up iteration for left base and right base for the lane lines.
3. stay with the window center if the valid pixels are greater then the threshold, if not then update to the value's mean to be the new base to start next rounds calculation.
4. if fitting has been down for previous frame, making use of the previous fitting parameters to narrrow down the searching area.
5. apply `numpy.polyfit()` to get fit params for indices on left and right lines.

*NOTE* several optimizations have been done to achieve smoother result, such as
1. `collections.deque(maxlen=5)` has been used to achieve a ring-buffer like behaviour, in order to buffer previous 4 frames' fitting result. In stead of update fit parameters direct to the line object, `Line.addFit()` shall be used, the fit param given would be a average result.
```python
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
```

2. instead of keeping 4 points of the searching window, a class of `Window` has been introduced, since 2 diagonal points would be good enough to identify a searching rectangle.
```python
#define window class to hold x, y, width, height position
#only 2 points are des
class Window:
    def __init__(self):
        self.left_top = None
        self.right_bottom = None      
```

the output example would be (source image as test2.jpg)

![alt text][imageLine]



#### 5. Measuring curvature and offset
With the finalize line indices from step4, in order to perform measurement on real-world. Assumption has been made from pixel to real-world meter as
```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

and the maximum value on pixel from the referenced image would be 719 ( since we always ensure the image is 1280 x 720 by shape). The formula to calculate the curvature and offset info has been adjusted into
```python
def measureCurvature(leftLine, rightLine):
    # Define conversions in x and y from pixels space to meters
    ymPerPix = 30/720 # meters per pixel in y dimension
    xmPerPix = 3.7/700 # meters per pixel in x dimension

    maxY = 719 # as it's determined to be 1280*720 by shape

    leftFitReal = np.polyfit(leftLine.all_y*ymPerPix, leftLine.all_x*xmPerPix, 2)
    rightFitReal = np.polyfit(rightLine.all_y*ymPerPix, rightLine.all_x*xmPerPix, 2)

    leftCur = pow((1+(2*leftFitReal[0]*maxY+leftFitReal[1])**2),1.5) / abs(2*leftFitReal[0])
    rightCur = pow((1+(2*rightFitReal[0]*maxY+rightFitReal[1])**2),1.5) / abs(2*rightFitReal[0])

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
```

with the above calucaltion the final output example would be (source image as test2.jpg)
![alt text][imageCur]


#### 6. Pipeline
To combine all the steps above, the finalized pipeline has been implemented in `lane_detection.imgFindLane()`. Including steps as __calibration__, __undistortion__, __thresholding__, __warp image__, __line fitting__, __finalize drawing__(*combine the finalized line fitting and project back to real world using inverse perspective matrix*).

![alt text][imageFinal]
+ *original image* as test_images/test2.jpg.
+ *target image* as same image after lane detection.

#### 7. Helper functions
In order to do ploting in a more abstracted way, several helper functions are introduced(mainly in `hmi.py`). Including:
* __plotImageWithCoordinateInfo__ plot images by using mpld3 plugins to get coordinates info
* __plotImages__ plot images with 2 subplots, tuple image info shall be provided (image, 'gray/None')
* __finalImgDrawing__ finalize drawing for lane detection



---

### Pipeline (video)

Here's a [link to my video result]( ./pipelineVideo/project_video_output.mp4)

Locally in `./pipelineVideo/project_video_output.mp4`


---

### Discussion
This project is only an initial trial for finding lane line, as the curvature info shown, it's not accurate enough to get a pretty result. Several problems I found during coding as:
1. __masking for binary images__ - definitely a double-edged sword. Since it's actually adding the limitation for curvature on the road. As I tried in other videos, it's sometimes not working at all. In real world situation, masking area shall be more carefully designed to order to fit more generic use cases.
2. __buffering logic__ - I added the buffering for 5 frames in order to achieve a more smoother result, but once it's reaching an error case, it would by taking more time to calibrate it. So I would say how much need to be done for buffering depends really on the output frequency of the camera.
3. __tuning algorithm__ - I tried combined thresholding for various filter, but personally I would say absolute sobel and color thresh seems to be more important and accurate in the given example. Binary operation by simply just `and` or `or` would be not enough if we actually want to add the value/coefficients to measure the weight would sounds more promising.
