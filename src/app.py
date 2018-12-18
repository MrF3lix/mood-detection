import sys
from DatasetPredictor import DatasetPredictor
from LandmarkDetector import LandmarkDetector

class App:
    def __init__(self):
        try:
            emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
            # emotions = ["neutral", "happy"]
            landmarkDetector = LandmarkDetector()
        
            self.datasetPredictor = DatasetPredictor(emotions,'models/emotion_evaluation_1.2_model.sav',landmarkDetector)
            self.datasetPredictor.Predict()

        except (KeyboardInterrupt, SystemExit):
            self.Stop()

    def Stop(self):
        self.datasetPredictor.Stop()


if __name__ == "__main__":
    App()
