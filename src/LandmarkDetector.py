import cv2
import numpy as np
import dlib
import math

class LandmarkDetector:
    def __init__(self):

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('targets/shape_predictor_68_face_landmarks.dat')

    def GetLandmarks(self, image):
        data = {}
        detections = self.detector(image, 1)

        for k,d in enumerate(detections):
            shape = self.predictor(image, d) 
            xlist = []
            ylist = []

            for i in range(1,68):
                cv2.circle(image, (shape.part(i).x,shape.part(i).y), 2, (0,0,255), -1)
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))

            xmean = np.mean(xlist)
            ymean = np.mean(ylist)
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

            data['landmarks_vectorised'] = landmarks_vectorised
        if len(detections) < 1:
            data['landmarks_vectorised'] = "error"

        return data
