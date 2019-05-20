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
    return img_blur

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
    # Perform morphology
    se = np.ones((7, 7), dtype='uint8')
    image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
    contours, hierarchy = cv2.findContours(image_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def drawContours(on,contours):
    mask = np.zeros(on.shape[:3], np.uint8)
    draw = cv2.drawContours(mask, contours, -1, (0,255,0), 1)
    drawn = cv2.bitwise_or(on, draw)
    return drawn

def scale(img):
    xs= img.shape[1]*1
    ys=img.shape[0]*1
    img_scale = cv2.resize(img, (int(xs),int(ys)), interpolation=cv2.INTER_AREA)
    return img_scale

def preProcessing(test):
    # load coloured image
    img_colour = read(test, 1)
    img_colour = scale(img_colour)
    display(img_colour, 'original image')

    # Load grayscale image
    img_gray = read(test, 0)
    img_gray = scale(img_gray)
    display(img_gray, 'grayscale image')

    # contrast stretching
    img_stretched = contrast_stretching(img_gray)
    display(img_stretched, 'stretched image')

    # Gaussian filtering/blur
    img_blur = gaussianBlur(img_stretched)
    display(img_blur, 'blurred image')

    return img_blur

def areaContour(contours,img_con):
    id = 0
    image = img_con
    for contour in contours:
        area = cv2.contourArea(contour)
        id = id+1
        # compute the center of the contour
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        image = cv2.putText(image, str(id), (cX,cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        print("Contour ", id, "  area ", area)

    if id==0:
        print("no contour")

    return image



#put input image here
test = 'net/test1.jpg'
preProcessed = preProcessing(test)


#########cracks###########

# Canny edge detection
img_edged = cannyEdge(preProcessed)
display(img_edged, 'edge detected image')

# Draw contour
contours = getContours(img_edged)
draw = read(test, 1)
draw = scale(draw)
img_con = drawContours(draw, contours)
display(img_con, 'detected cracks')

#######chips###########

'''
# for chips
areaContour(contours, img_con)
display(img_con, 'areas')
'''

#differentite crack and chip
'''se = np.ones((7, 7), dtype='uint8')
image_close = cv2.morphologyEx(img_filled, cv2.MORPH_CLOSE, se)
contours, hierarchy = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_ex = drawContours(img_filled, read(test, 1),contours)
display(img_ex, 'external boundry')'''

'''

co=0
for c in contours:
    if cv2.contourArea(c)>100:
        co=co+1
        print ("co",co)
        hull = cv2.convexHull(c, returnPoints=False)
        defects = cv2.convexityDefects(c, hull)

        if defects is not  None:
            line=0
            cir=0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(c[s][0])
                end = tuple(c[e][0])
                far = tuple(c[f][0])
                cv2.line(img_con, start, end, [0, 255, 0], 1)
                line=line+1
                print ("line", line)
                cv2.circle(img_con, far, 2, [0, 0, 255], -1)
                cir=cir+1
                print ("circle", cir)





cv2.imshow('hull',img_con)
cv2.waitKey(0)

'''

