import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

video = cv.VideoCapture('video/dashcam01.mp4')

"""
    NOTES:
    Press 'Q' to stop the program.
"""

class DisplayHelper():
    def resizeImg(self, img, new_width):
        resized_img = img
        if img is not None:
            height, width = img.shape[:2]
            if height > 0 and width > 0:
                aspect_ratio = height / width
                new_height = int(new_width * aspect_ratio)
                resized_img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)
        return resized_img

    def overlayImgsWithAlpha(self, img1, img2, alpha=0.5):
        beta = 1.0 - alpha
        overlayed = cv.addWeighted(img1, alpha, img2, beta, 0.0)
        return overlayed

    def captureFramesBySeconds(self, video_src, n_frames, delta_t=0):
        captured_frames = []
        first_t = datetime.now().second
        counter = 0
        while counter < n_frames:
            rval, frame = video_src.read()
            if delta_t > 0:
                curr_t = datetime.now().second
                if counter == 0:
                    first_t = curr_t
                    captured_frames.append(frame)
                    counter += 1
                elif counter > 0 and curr_t == (first_t + delta_t):
                    first_t = curr_t
                    captured_frames.append(frame)
                    counter += 1
            else:
                captured_frames.append(frame)
                counter += 1
        return captured_frames

class CustomExtractor():
    def __init__(self):
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        self.last_img = None

    def getKps(self, img):
        kps = []
        img_blurred = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
        img_gray = cv.cvtColor(img_blurred, cv.COLOR_BGR2GRAY)
        img_canny = cv.Canny(img_gray, 100, 200)
        contours, hierarchy = cv.findContours(img_canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            kps.append(cv.KeyPoint(x=float(c[0][0][0]), y=float(c[0][0][1]), size=20))
        return kps

    def getDescriptors(self, img, kps):
        extractor = cv.xfeatures2d.BriefDescriptorExtractor_create()
        kps, desc = extractor.compute(img, kps)
        return desc

    def drawKps(self, img, kps):
        res = None
        for k in kps:
            res = cv.circle(img, (int(k.pt[0]), int(k.pt[1])), 4, (0, 255, 0), -1)
        return res

    def getFeatures(self, img):
        kps = self.getKps(img)
        desc = self.getDescriptors(img, kps)
        return {'kps': kps, 'desc': desc}

class ORBExtractorImpl():
    def __init__(self):
        self.feats_n = 2000
        self.bfmatcher = cv.BFMatcher()
        self.orb = cv.ORB_create()
        self.last_img = None

    def extract(self, img):
        img_out = img
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_feats = cv.goodFeaturesToTrack(img_gray, self.feats_n, qualityLevel=0.01, minDistance=1)
        if(img_feats is not None) and (not np.all(img == 0)):
            kps = []
            for f in img_feats:
                kps.append(cv.KeyPoint(x=f[0][0], y=f[0][1], size=20))
            kps, desc = self.orb.compute(img, kps)
            img_out = cv.drawKeypoints(img, kps, None, (0, 255, 0), flags=0)

            if self.last_img is None:
                self.last_img = {'img': img, 'kps': kps, 'desc': desc}
                return img_out
            else:
                # Query image = last image
                # Train image = current image
                knn_matches = self.bfmatcher.knnMatch(self.last_img['desc'], desc, k=2)
                good_knn_matches = []

                query_img_pts = []
                train_img_pts = []
                for m, n in knn_matches:
                    if m.distance < 0.65*n.distance:
                        good_knn_matches.append(m)
                for m in good_knn_matches:
                    query_kp = (int(self.last_img['kps'][m.queryIdx].pt[0]), int(self.last_img['kps'][m.queryIdx].pt[1]))
                    train_kp = (int(kps[m.trainIdx].pt[0]), int(kps[m.trainIdx].pt[1]))
                    cv.line(img_out, query_kp, train_kp, (255, 0, 0), 1)
                    query_img_pts.append(list(self.last_img['kps'][m.queryIdx].pt))
                    train_img_pts.append(list(kps[m.trainIdx].pt))
                
                query_img_pts = np.array(query_img_pts).astype(np.uint8)
                train_img_pts = np.array(train_img_pts).astype(np.uint8)
                homography, mask = cv.findHomography(query_img_pts, train_img_pts, cv.RANSAC, ransacReprojThreshold=2.0)
                print(homography)
                self.last_img = {'img': img, 'kps': kps, 'desc': desc}
                return img_out
        else:
            return img_out

Extractor = CustomExtractor()
ORBExtractor = ORBExtractorImpl()
DisplayHelper = DisplayHelper()

# Play the video
while True:
    if video.isOpened():
        isTrue, frame = video.read()

        if frame is None:
            break

        """
        CUSTOM FEATURE DETECTION AND KEYPOINTS EXTRACTION
        """
        frame_features = Extractor.getFeatures(frame)
        # frame_with_kps = Extractor.drawKps(frame, frame_features['kps'])
        frame_with_kps = cv.drawKeypoints(frame, frame_features['kps'], None, (0, 255, 0), flags=0)
        video_fps = 'FPS({fps})'.format(fps=int(video.get(cv.CAP_PROP_FPS)))
        cv.putText(frame_with_kps, video_fps, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if frame_with_kps is not None:
            cv.imshow('frame keypoints', DisplayHelper.resizeImg(frame_with_kps, 900))

        """
        OPENCV ORB FEATURE DETECTION AND KEYPOINTS EXTRACTION
        """
        # extracted_img = ORBExtractor.extract(frame)
        # cv.imshow('extracted', DisplayHelper.resizeImg(extracted_img, 1200))

        if cv.waitKey(20) & 0xFF==ord('q'):
            break

# captured_frames = DisplayHelper.captureFramesBySeconds(video, 2, 1)
# for indx, f in enumerate(captured_frames):
#     """
#     OPENCV ORB IMPLEMENTATION
#     """
#     extracted_img = ORBExtractor.extract(f)
#     cv.imshow(f'orbs {indx}', DisplayHelper.resizeImg(extracted_img, 900))

    # """
    # CUSTOM IMPLEMENTATION
    # """
    # f_features = Extractor.getFeatures(f)
    # f_kps_img = Extractor.drawKps(f, f_features['kps'])
    # cv.imshow(f'frame n{indx} kps', DisplayHelper.resizeImg(f_kps_img, 900))

# plt.imshow(cv.cvtColor(test_img, cv.COLOR_BGR2RGB))
# plt.show()

# cv.waitKey(0)
video.release()
cv.destroyAllWindows()