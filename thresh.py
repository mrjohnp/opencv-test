import cv2 as cv

img = cv.imread('img/city.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Image', img)

# Thresholding: binarizing an image
# an image where pixel are either 0 (black) or 255 (white)

# Simple Thresholding
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Simple Thresholded', thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Simple Thresholded Inverse', thresh_inv)

# Adaptive Thresholding
# it finds the optimal threshold value
adaptive_thresh = cv.adaptiveThreshold(
    gray, 255,
    cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
    11, 3)

cv.imshow('Adaptive Thresholding', adaptive_thresh)

cv.waitKey(0)