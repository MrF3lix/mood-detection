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
        try:
            while True:
                _, frame = self.cam.read()
                frame = cv2.resize(frame, (960, 540)) 
                # self.landmarkDetector.ShowLandmarks(frame)


                test = cv2.resize(frame, (384, 216)) 
                # TODO make sure the frame only takes the face and is 350 by 350 pixel
                landmarks = self.landmarkDetector.GetLandmarks(test)

                if landmarks != "error":
                    
                    prediction = self.clf.predict_proba(landmarks)

                    print('--------\nPrediction')
                    print(prediction)
                    for n in range(0, len(prediction)):
                        for i in range(0,len(self.classList)):

                            value = float(prediction[n][i] * 100)
                            print('%s: %s' %(self.classList[i], round(value)))

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