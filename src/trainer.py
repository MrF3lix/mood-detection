import sys
from DatasetTrainer import DatasetTrainer
from LandmarkDetector import LandmarkDetector

class Trainer:
    def __init__(self):
        self.emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
        landmarkDetector = LandmarkDetector()
        datasetTrainer = DatasetTrainer(self.emotions, landmarkDetector)

        datasetTrainer.Train()

if __name__ == "__main__":
    Trainer()
