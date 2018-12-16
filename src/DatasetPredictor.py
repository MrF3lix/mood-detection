import cv2
import numpy as np
import dlib
import pickle

class DatasetPredictor:
    def __init__(self, classList,model,landmarkDetector):
        self.landmarkDetector = landmarkDetector
        self.classList = classList
        self.cam = cv2.VideoCapture(0)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('targets/shape_predictor_68_face_landmarks.dat')

        self.clf = pickle.load(open(model, 'rb'))

    def Predict(self):
        try:
            while True:
                faces = []
                self.data = {}
                _, frame = self.cam.read()

                data = self.landmarkDetector.GetLandmarks(frame)
                landmarks = data['landmarks_vectorised']

                if landmarks != "error":
                    npar_pred = np.array(landmarks)
                    faces.append(npar_pred)
                    
                    prediction = self.clf.predict_proba(faces)

                    print('--------\nPrediction')
                    for n in range(0, len(prediction)):
                        for i in range(0,5):
                            print('%s: %s' %(self.classList[i], prediction[n][i]))

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