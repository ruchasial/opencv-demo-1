import cv2
import numpy as np
import scipy.ndimage

test = 'test1.jpg'

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

'''#gaussian blur- original
img_sub=cv2.subtract(img_blur,img_contStretched)

cv2.imshow('Gaussian blur- original', img_sub)
cv2.waitKey(0)'''
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


contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

'''maxarea=0
for c in contours:
    area= cv2.contourArea(c)
    if area > maxarea:
        maxarea = area
        maxcont = c'''

img_con=cv2.drawContours(img, contours, -1, (0,255,0), -1)
cv2.imshow('contours', img_con)
cv2.waitKey(0)


print("lengths are - \n")
for c in contours:
    p = cv2.arcLength(c, False)
    print (p)











