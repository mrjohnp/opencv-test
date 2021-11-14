import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('img/city.jpg')
cv.imshow('Image', img)

# Histogram: allows to visualize the pixel intensity on an image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# Gray histogram
# Shows frequency (how many times) each gray level occurs
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])

# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('n of pixel')
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.show()

# NOTE: apply a MASK to compute a histogram of particular section of the image

# Colour Histogram
# Show the intensity of color channels
plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('n of pixel')
colors = ('b', 'g', 'r')

for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()

cv.waitKey(0)