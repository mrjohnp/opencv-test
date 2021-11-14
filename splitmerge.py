import cv2 as cv
import numpy as np

img = cv.imread('img/city.jpg')
cv.imshow('City', img)

# Splits the image into blue green and red
b,g,r = cv.split(img)

# Show the concentration of the color in grayscale
# brighter >> more concentrated
# darker >> least concentrated or not present at all
# cv.imshow('blue', b)
# cv.imshow('green', g)
# cv.imshow('red', r)

# (n1, n2, n3)
# n3 > number of color channels
print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

# Merge the individual color channel
merged = cv.merge([b,g,r])
# cv.imshow('Merged', merged)

# Display only a specific color channel
blank = np.zeros(img.shape[:2], dtype='uint8')
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

cv.waitKey(0)
