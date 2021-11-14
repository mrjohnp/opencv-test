import cv2 as cv

img = cv.imread('img/city.jpg')
cv.imshow('City', img)

# Averaging blur
# Apply the average pixel intensity of
# the sorrounding pixels in the kernel window
average = cv.blur(img, (3,3))
cv.imshow('Average', average)

# Gaussian blur
guass = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gaussian', guass)

# Median blur
# Apply the median intensity of the sorrounding pixels
# It reduce a certain amount of noise
median = cv.medianBlur(img, 3)
cv.imshow('Median', median)

# Bilateral blur
# Most effective
# It retains the edges of the image unlike other blurring methods
bilateral = cv.bilateralFilter(img, 7, 15, 15)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)