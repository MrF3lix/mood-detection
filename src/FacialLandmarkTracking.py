import imutils
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import dlib
import math
import cv2
import numpy as np

class FacialLandmarkTracking:
    def __init__(self):
        # You can use multiple cameras
        self.cam = cv2.VideoCapture(1)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('targets/shape_predictor_68_face_landmarks.dat')

        cv2.namedWindow('Webcam', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Webcam', 0, 0)
        cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)


    def OutputWebcamFrame(self):
        _, frame = self.cam.read()
        landmarks = self.GetLandmarks(frame)


        # TODO keep track of those landmarks and detect changes
        # TODO evaluate emotions from landmarks on the face

        cv2.imshow('Webcam', frame)

    def GetLandmarks(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(frame_gray, 0)
        for face in faces:
            shape = self.predictor(frame_gray, face)
            shape = face_utils.shape_to_np(shape)

            xlist = []
            ylist = []

            for (x,y) in shape:
                cv2.circle(frame, (x,y), 2, (0,0,255), -1)

                xlist.append(float(x))
                ylist.append(float(y))

            # Get center of gravity
            xmean = np.mean(xlist)
            ymean = np.mean(ylist) 

            # Get distance from point to center of gravity
            xcentral = [(x-xmean) for x in xlist] 
            ycentral = [(y-ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))

        if len(faces) > 0:
            return landmarks_vectorised
        else:
            landmarks = "error"
            return landmarks_vectorised

    def Stop(self):
        self.cam.release()
        cv2.destroyAllWindows()