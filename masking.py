import cv2 as cv
import numpy as np

img = cv.imread('img/city.jpg')
cv.imshow('City', img)

# NB: The mask has to be the same dimension of the image
# otherwise is gonna give an error
blank = np.zeros(img.shape[:2], dtype='uint8')

mask_circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
masked_circle = cv.bitwise_and(img, img, mask=mask_circle)
cv.imshow('Masked - circle', masked_circle)

cv.waitKey(0)