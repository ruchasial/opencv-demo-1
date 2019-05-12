import cv2
import numpy as np
from scipy import ndimage
from skimage.exposure import rescale_intensity
# from matplotlib import pyplot as plt


test = 'test.jpg'

####### load coloured image##########
img = cv2.imread(test, 1)
cv2.imshow('original image', img)
cv2.waitKey(0)
#cv2.destroyWindows()



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


###########image segmentation##########
'''
img_blur = cv2.medianBlur(img_contStretched, 5)
cv2.imshow('median blur', img_blur)
cv2.waitKey(0)
img_segmented = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('segmented image', img_segmented)
cv2.waitKey(0) '''

 #Gaussian filtering
img_blur = cv2.GaussianBlur(img_contStretched, (5, 5), 0)
cv2.imshow('Gaussian blur', img_blur)
cv2.waitKey(0)

'''#thresholding
ret, img_segmented = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)
#ret, img_segmented = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('segmented image', img_segmented)
cv2.waitKey(0)'''

#thresholding


#edge detection
kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
img_segmented = ndimage.convolve(img_blur, kernel_laplace, mode='reflect')
cv2.imshow('segmented image', img_segmented)
cv2.waitKey(0)


###########morphological operations##########

def bwmorphClean(image):
    (iH, iW) = image.shape[:2]
    pad = 1
    image = cv2.copyMakeBorder(image, pad, pad,  pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            if roi[0, 0] == roi[1, 0] == roi[2, 0] == roi[0, 1] == roi[0, 2] == roi [1,2] == roi [2,1] == roi [2,2] == 0:
                output[y - pad, x - pad] = 0
            else:
                output[y - pad, x - pad] = roi[1,1]

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output

img_cleaned=bwmorphClean(img_segmented)
cv2.imshow('cleaned', img_cleaned)
cv2.waitKey(0)


# Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)
img_dilation = cv2.dilate(img_segmented, kernel, iterations=1)
cv2.imshow('dilated', img_dilation)
cv2.waitKey(0)

img_erosion = cv2.erode(img_segmented, kernel, iterations=1)
cv2.imshow('eroded', img_erosion)
cv2.waitKey(0)
#img_fill = cv2.fill



