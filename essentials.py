import cv2 as cv

img = cv.imread('img/toucan.jpg')

def grayScale(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)

def blur(src):
    # To increase the "blurriness", change che kernel-size tuple
    # to a higher value > eg. (13, 13)
    blurred = cv.GaussianBlur(src, (7, 7), cv.BORDER_DEFAULT)
    cv.imshow('Blur', blurred)

def edgeCascade(src):
    # it uses the canny edge detector
    canny = cv.Canny(src, 125, 175)
    # to reduce the edges, pass a blurred image
    less_edges = cv.Canny(cv.GaussianBlur(src, (7, 7), cv.BORDER_DEFAULT), 125, 175)

    cv.imshow('Canny', canny)
    cv.imshow('Canny - less edges', less_edges)

def dilate(src):
    dialated = cv.dilate(src, (11, 11), iterations=6)
    cv.imshow('Dialated', dialated)

def erode(src):
    eroded = cv.erode(cv.dilate(src, (11, 11), iterations=6), (11, 11), iterations=6)
    cv.imshow('Eroded', eroded)

def resize(src):
    # This will not keep the aspect ratio of the image
    # interpolation= cv.INT_AREA > is usefull if the image shrinks
    # interpolation= cv.INT_LINEAR or cv.INT_CUBIC > are usefull if the image grows
    resized = cv.resize(src, (500, 500), interpolation=cv.INTER_AREA)
    cv.imshow('Resized', resized)

def crop(src):
    # The image is essentially an array of pixelsz
    cropped = src[100:200, 100:300]
    cv.imshow('Cropped', cropped)

# grayScale(img)
# blur(img)
# edgeCascade(img)
# dilate(img)
# erode(img)
# resize(img)
crop(img)
cv.imshow('Original', img)
cv.waitKey(0)