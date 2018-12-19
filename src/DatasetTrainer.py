import glob
from shutil import copyfile
import os
import cv2
import random
import numpy as np
import dlib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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

        data = train_test_split(files, test_size=0.33, random_state=42)

        training = data[0]
        prediction = data[1]

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
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                landmarks = self.landmarkDetector.GetLandmarks(image_gray)
                if landmarks == "error":
                    print("failed to detect a face: %s" %(item))
                else:
                    print("training: %s" %(item))
                    training_data.extend(landmarks)
                    training_labels.append(self.classList.index(emotion))

            for item in prediction:
                image = cv2.imread(item)
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                landmarks = self.landmarkDetector.GetLandmarks(image_gray)
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
        for i in range(0,10):
            print('Training run: %s' %(i))
            result = self.Run()
            accur_lin.append(result)

        print("Mean value lin svm: %s" %np.mean(accur_lin))

        filename = 'models/emotion_evaluation_2.0_model.sav'
        pickle.dump(self.clf, open(filename, 'wb'))