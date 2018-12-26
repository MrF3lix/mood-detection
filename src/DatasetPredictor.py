import cv2
import numpy as np
import dlib
import pickle
import math

class DatasetPredictor:
    def __init__(self,classList,model,landmarkDetector):
        self.landmarkDetector = landmarkDetector
        self.classList = classList
        self.cam = cv2.VideoCapture(1)

        self.clf = pickle.load(open(model, 'rb'))

    def Predict(self):
        face_detect = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

        try:
            while True:

                face_landmarks = []

                _, frame = self.cam.read()

                frame = cv2.resize(frame, (480,270))

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = face_detect.detectMultiScale(frame_gray, 1.1, 5)

                for (x, y, w, h) in face:
                    face = frame_gray[y:y+h, x:x+w] 
                    face_resized = cv2.resize(face, (350,350))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    landmarks = self.landmarkDetector.GetLandmarks(face_resized)

                    if landmarks != "error":
                        face_landmarks.append(landmarks)

                        
                if len(face_landmarks) > 0:
                    prediction = self.clf.predict_proba(face_landmarks)

                    print('--------\nPrediction')
                    for n in range(0, len(prediction)):
                        for i in range(0,len(self.classList)):

                            value = float(prediction[n][i] * 100)
                            print('Face-%s: %s: %s' %(n, self.classList[i], round(value)))

                cv2.imshow('Webcam', frame)

                k = cv2.waitKey(1)
                if k % 256 == 27:
                    self.Stop()
                    break

        except (KeyboardInterrupt, SystemExit):
            self.Stop()

    def Stop(self):
        print('Exit application')
        self.cam.release()
        cv2.destroyAllWindows()