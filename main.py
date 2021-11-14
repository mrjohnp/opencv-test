import cv2 as cv
import numpy as np

def showImg(path):
    # Read an image an convert it into a matrix
    img = cv.imread(path)
    # Show the img into a window
    cv.imshow('Image', img)
    # Wait any key to be pressed to close the window
    cv.waitKey(0)

def showVideo(path):
    # Read the video
    capture = cv.VideoCapture(path)

    # Read the video frame by frame
    while True:
        # Read the frame
        # isTrue: boolean that says wheter the frame is successfully read or not
        isTrue, frame = capture.read()
        # Show the frame into the window
        cv.imshow('Video', frame)

        # Exit if "d" key is pressed
        if cv.waitKey(20) & 0xFF==ord('d'):
            break

    # Release capture pointer
    capture.release()
    # Destroy window
    cv.destroyAllWindows()

def rescaleFrame(frame, scale=0.75):
    # Get frame's width
    width = int(frame.shape[1] * scale)
    # Get frame's height
    height = int(frame.shape[0] * scale)
    # Save the 2 dimenions in a tuple
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# This only works on live video
def changeRes(capture, width, height):
    # 3 refers to the width property of the capture object
    capture.set(3, width)
    # 4 refers to the height property of the capture object
    capture.set(4, height)
    return capture

# This works on image, video and live video
def showResizedVideo(path, scale):
    # Read the video
    capture = cv.VideoCapture(path)

    # Read the video frame by frame
    while True:
        # Read the frame
        # isTrue: boolean that says wheter the frame is successfully read or not
        isTrue, frame = capture.read()
        frame_resized = rescaleFrame(frame, scale)
        # Show the frame into the window
        cv.imshow('Video resized', frame_resized)

        # Exit if "d" key is pressed
        if cv.waitKey(20) & 0xFF==ord('d'):
            break

    # Release capture pointer
    capture.release()
    # Destroy window
    cv.destroyAllWindows()
