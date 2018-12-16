import glob
from shutil import copyfile
import os
import cv2
import random
import numpy as np
import dlib
from sklearn.svm import SVC
import math
import pickle

class DatasetTrainer:
    def __init__(self, classList, landmarkDetector):
        self.classList = classList
        self.landmarkDetector = landmarkDetector

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('targets/shape_predictor_68_face_landmarks.dat')
        self.clf = SVC(kernel='linear', probability=True, tol=1e-3)

    def GetTrainingFiles(self, emotion):
        files = glob.glob("dataset/faces/%s/*" %emotion)
        random.shuffle(files)

        training = files[:int(len(files)*0.8)]
        prediction = files[-int(len(files)*0.2):]

        return training, prediction

    def InitTrainingSet(self):
        training_data = []
        training_labels = []
        prediction_data = []
        prediction_labels = []

        for emotion in self.classList:
            training, prediction = self.GetTrainingFiles(emotion)

            for item in training:
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe_image = self.clahe.apply(gray)

                landmarks = self.landmarkDetector.GetLandmarks(clahe_image)
                if landmarks['landmarks_vectorised'] == "error":
                    print("failed to detect a face: %s" %(item))
                else:
                    training_data.append(landmarks['landmarks_vectorised'])
                    training_labels.append(self.classList.index(emotion))

            for item in prediction:
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe_image = self.clahe.apply(gray)

                landmarks = self.landmarkDetector.GetLandmarks(clahe_image)
                if landmarks['landmarks_vectorised'] == "error":
                    print("failed to detect a face: %s" %(item))
                else:
                    prediction_data.append(landmarks['landmarks_vectorised'])
                    prediction_labels.append(self.classList.index(emotion))

        return training_data, training_labels, prediction_data, prediction_labels

    def Run(self):
        training_data, training_labels, prediction_data, prediction_labels = self.InitTrainingSet()

        # Train the model
        npar_train = np.array(training_data)
        self.clf.fit(npar_train, training_labels)

        # Test the model 
        npar_pred = np.array(prediction_data)
        pred_lin = self.clf.score(npar_pred, prediction_labels)

        return pred_lin

    def Train(self):
        accur_lin = []
        for i in range(0,10):
            print('Training run: %s' %(i))
            result = self.Run()
            accur_lin.append(result)

        print("Mean value lin svm: %s" %np.mean(accur_lin))

        filename = 'models/emotion_evaluation_1.0_model.sav'
        pickle.dump(self.clf, open(filename, 'wb'))