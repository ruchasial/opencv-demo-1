import cv2
import numpy as np
import scipy.ndimage

test = 'test1.jpg'

def display (image, desc):
    cv2.imshow(desc, image)
    cv2.waitKey(0)

def skeletonize(gray_img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = gray_img.copy() # don't clobber original
    skel = gray_img.copy()
    print("1")

    skel[:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    print("2")


    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(gray_img, temp)
        print("3")
        skel = cv2.bitwise_or(skel, temp)
        img [:,:] = eroded[:,:]
        if cv2.countNonZero(eroded) == 0:
           break

    return skel

####### load coloured image##########

img = cv2.imread(test, 1)
cv2.imshow('original image', img)
cv2.waitKey(0)

###########convert image to grayscale##########

img_gray = cv2.imread(test, 0)
cv2.imshow('grayscale', img_gray)
cv2.waitKey(0)

########### contrast stretching##########

# Create zeros array to store the stretched image
img_contStretched = np.zeros((img_gray.shape[0], img_gray.shape[1]), dtype='uint8')

# Loop over the image and apply Min-Max formulae
for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        img_contStretched[i, j] = 255 * (img_gray[i, j] - np.min(img_gray)) / (np.max(img_gray) - np.min(img_gray))

# Display the stretched image
cv2.imshow('contrast stretched', img_contStretched)
cv2.waitKey(0)

########Gaussian filtering###########

#gaussian blur
img_blur = cv2.GaussianBlur(img_contStretched, (5, 5), 0)
cv2.imshow('Gaussian blur', img_blur)
cv2.waitKey(0)
img_sub = img_blur

########edge detection ############

# compute the median of the single channel pixel intensities
v = np.median(img_sub)

# apply automatic Canny edge detection using the computed median
sigma = 0.33
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
img_edged = cv2.Canny(img_sub, lower, upper)
cv2.imshow('canny edged', img_edged)
cv2.waitKey(0)


#Close the contours
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
dilated = cv2.dilate(img_edged, kernel)
eroded = cv2.erode(dilated, kernel)
cv2.imshow('closed edged', eroded)
cv2.waitKey(0)

'''
img_skel=skeletonize(eroded)
cv2.imshow('skeleton edged', img_skel)
cv2.waitKey(0)'''

contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_con=cv2.drawContours(img, contours, -1, (0,255,0), -1)
cv2.imshow('contours', img_con)
cv2.waitKey(0)





'''
print("lengths are - \n")
for c in contours:
    p = cv2.arcLength(c, False)
    print (p)'''










