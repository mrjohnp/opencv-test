import cv2 as cv
import numpy as np

img = cv.imread('img/city.jpg')
cv.imshow('Image', img)

# Blank image that has the size of the image
blank = np.zeros(img.shape, dtype='uint8')
# cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)

# Pass a blurred image to decrease the number of contours
canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny', canny)

# Thresholding an image: binarize the image
# eg. (125, 255)
# if the pixel density is below 125 is gonna be set to 0 so black
# if the pixel is above 255 is gonna be 255 so white
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

# Find the number of contours are in the image
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found')

# Draw contours on blank image
cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Draw Contours', blank)

cv.waitKey(0)