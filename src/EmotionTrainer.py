import glob
from shutil import copyfile
import os
import cv2
import random
import numpy as np

class EmotionTrainer:
    def __init__(self):
        # self.emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
        self.emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
        self.fisher_face_reco = cv2.face.FisherFaceRecognizer_create()
        self.data = {}
        self.cam = cv2.VideoCapture(1)

        self.face_detect = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
        # self.CleanDataset()
        # self.CollectFaces()

    def CleanDataset(self):
        participants = glob.glob('dataset/source_emotion/*', recursive=True)
        for x in participants:
            part = '%s' %x[-4:]

            for sessions in glob.glob('%s/*' %x): 
                for files in glob.glob('%s/*' %sessions):
                    current_session = files[28:-30]
                    file = open(files, 'r')
                    emotion = int(float(file.readline()))

                    if(len(self.emotions) - 1 < emotion):
                        continue

                    source_files = sorted(glob.glob('dataset/source_image/%s/%s/*' %(part, current_session)))

                    source_emotion = source_files[-1]
                    source_neutral = source_files[0]

                    copyfile(source_neutral, 'dataset/sorted_set/%s/%s' %('neutral', source_neutral[30:]))
                    copyfile(source_emotion, 'dataset/sorted_set/%s/%s' %(self.emotions[emotion], source_emotion[30:]))
                    print('copied')
        

    def CollectFaces(self):
        face_detect = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

        for emotion in self.emotions:
            files = glob.glob('dataset/sorted_set/%s/*' %(emotion))
            file_index = 0

            for f in files:
                frame = cv2.imread(f)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = face_detect.detectMultiScale(frame_gray, 1.1, 5)

                for (x, y, w, h) in face:
                    frame_gray = frame_gray[y:y+h, x:x+w] 
                    try:
                        print('found face')
                        frame_out = cv2.resize(frame_gray, (350, 350))
                        cv2.imwrite('dataset/faces/%s/%s.jpg' %(emotion, file_index), frame_out)
                        file_index += 1
                    except:
                        print('failed')
                        pass

    def GetTrainingFiles(self, emotion):
        files = glob.glob("dataset/faces/%s/*" %emotion)
        random.shuffle(files)
        training = files[:int(len(files)*0.8)] #get first 80% of file list
        prediction = files[-int(len(files)*0.2):] #get last 20% of file list

        return training, prediction

    def InitTrainingSet(self):
        training_data = []
        training_labels = []
        prediction_data = []
        prediction_labels = []
        for emotion in self.emotions:
            training, prediction = self.GetTrainingFiles(emotion)
            #Append data to training and prediction list, and generate labels 0-7

            for item in training:
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                training_data.append(gray)
                training_labels.append(self.emotions.index(emotion))

            for item in prediction:
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                prediction_data.append(gray)
                prediction_labels.append(self.emotions.index(emotion))
        return training_data, training_labels, prediction_data, prediction_labels

    def RunRecognizer(self):
        training_data, training_labels, prediction_data, prediction_labels = self.InitTrainingSet()

        self.fisher_face_reco.train(training_data, np.asarray(training_labels))
        cnt = 0
        correct = 0
        incorrect = 0
        for image in prediction_data:
            pred, conf = self.fisher_face_reco.predict(image)
            if pred == prediction_labels[cnt]:
                correct += 1
                cnt += 1
            else:
                cv2.imwrite("dataset/difficult/%s_%s_%s.jpg" %(self.emotions[prediction_labels[cnt]], self.emotions[pred], cnt), image) 
                incorrect += 1
                cnt += 1

        return ((100*correct)/(correct + incorrect))

    def Train(self):
        metascore = []
        for i in range(0,1):
            print('Training run:')
            print(i)
            result = self.RunRecognizer()
            metascore.append(result)

        print('\nEnd score: %s percent correct!' %(np.mean(metascore)))

    def Predict(self):
        try:
            while True:

                # Collect image from webcam and cut down
                _, frame = self.cam.read()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = self.face_detect.detectMultiScale(frame_gray, 1.1, 5)

                for (x, y, w, h) in face:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

                    frame_gray = frame_gray[y:y+h, x:x+w] 
                    frame_out = cv2.resize(frame_gray, (350, 350))
                    pred, conf = self.fisher_face_reco.predict(frame_out)
                    print('prediction %s, confidence: %s' %(self.emotions[pred], conf))



                cv2.imshow('Webcam', frame)

                k = cv2.waitKey(1)
                if k % 256 == 27:
                    self.Stop()
                    break

        except (KeyboardInterrupt, SystemExit):
            self.Stop()


    def Stop(self):
        print('stop')