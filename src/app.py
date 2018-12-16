import sys
from FacialLandmarkTracking import FacialLandmarkTracking
from EmotionTrainer import EmotionTrainer
from DatasetPredictor import DatasetPredictor
from LandmarkDetector import LandmarkDetector

class App:
    def __init__(self):
        try:
            self.emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
            landmarkDetector = LandmarkDetector()
        
            self.datasetPredictor = DatasetPredictor(self.emotions,'models/emotion_evaluation_1.0_model.sav',landmarkDetector)
            self.datasetPredictor.Predict()

        except (KeyboardInterrupt, SystemExit):
            self.Stop()

    def Stop(self):
        self.datasetPredictor.Stop()


if __name__ == "__main__":
    App()
