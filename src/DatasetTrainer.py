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
                landmarks = self.landmarkDetector.GetLandmarks(image)
                if landmarks == "error":
                    print("failed to detect a face: %s" %(item))
                else:
                    print("training: %s" %(item))
                    training_data.extend(landmarks)
                    training_labels.append(self.classList.index(emotion))

            for item in prediction:
                image = cv2.imread(item)
                landmarks = self.landmarkDetector.GetLandmarks(image)
                if landmarks == "error":
                    print("failed to detect a face: %s" %(item))
                else:
                    print("predicting: %s" %(item))
                    prediction_data.extend(landmarks)
                    prediction_labels.append(self.classList.index(emotion))

        return training_data, training_labels, prediction_data, prediction_labels

    def Run(self):
        training_data, training_labels, prediction_data, prediction_labels = self.InitTrainingSet()

        # Train the model
        self.clf.fit(np.array(training_data), training_labels)

        # Test the model 
        pred_lin = self.clf.score(np.array(prediction_data), prediction_labels)

        return pred_lin

    def Train(self):
        accur_lin = []
        for i in range(0,1):
            print('Training run: %s' %(i))
            result = self.Run()
            accur_lin.append(result)

        print("Mean value lin svm: %s" %np.mean(accur_lin))

        filename = 'models/emotion_evaluation_1.1_model.sav'
        pickle.dump(self.clf, open(filename, 'wb'))