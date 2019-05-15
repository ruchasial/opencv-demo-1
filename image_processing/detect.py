import cv2
import numpy as np
import scipy.ndimage

def display (image, desc):
    cv2.imshow(desc, image)
    cv2.waitKey(0)

def read(image, param):

    img_read = cv2.imread(image, param)
    return img_read

def contrast_stretching (image):
    # Create zeros array to store the stretched image
    img_contStretched = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')

    # Loop over the image and apply Min-Max formulae
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_contStretched[i, j] = 255 * (image[i, j] - np.min(image)) / (np.max(image) - np.min(image))

    return  img_contStretched

def gaussianBlur(image):

    img_blur = cv2.GaussianBlur(image, (5, 5), 0)
    return  img_blur

def cannyEdge (image):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    img_edged = cv2.Canny(image, lower, upper)
    return img_edged


#dilate + erode image contours
def fillContours(image):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(img_edged, kernel)
    eroded = cv2.erode(dilated, kernel)
    return eroded

def getContours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def drawContours(image,on,contours):
    drawn = cv2.drawContours(on, contours, -1, (0, 255, 0), -1)
    return  drawn

def preProcessing(test):
    # load coloured image
    img_colour = read(test, 1)
    display(img_colour, 'original image')

    # Load grayscale image
    img_gray = read(test, 0)
    display(img_gray, 'grayscale image')

    # contrast stretching
    img_stretched = contrast_stretching(img_gray)
    display(img_stretched, 'stretched image')

    # Gaussian filtering/blur
    img_blur = gaussianBlur(img_stretched)
    display(img_blur, 'blurred image')

    return img_blur


test = 'test1.jpg'
preProcessed = preProcessing(test)

# Canny edge detection
img_edged = cannyEdge(preProcessed)
display(img_edged, 'edge detected image')

# Close/fill the contours
img_filled = fillContours(img_edged)
display(img_filled, 'filled contours')

# Draw contour
contours=getContours(img_filled)
img_con = drawContours(img_filled, read(test, 1),contours)
display(img_con, 'detected cracks')


#Find the index of the largest contour
if len(contours) != 0:
    #find the biggest area
    c = max(contours, key = cv2.contourArea)

    x,y,w,h = cv2.boundingRect(c)
    # draw the book contour (in green)
    max_cont=cv2.rectangle(read(test,1),(x,y),(x+w,y+h),(0,255,0),2)

    display(max_cont,'largest chip')

