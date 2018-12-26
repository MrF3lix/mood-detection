import sys
import argparse
from DatasetTrainer import DatasetTrainer
from DatasetPredictor import DatasetPredictor
from LandmarkDetector import LandmarkDetector
from DatasetPrep import DatasetPrep

parser = argparse.ArgumentParser(description="Emotion recognition app.")
parser.add_argument('--task', type=str, help='Defines the task that should be executed (train,predict)')

class App:
    def __init__(self):
        args = parser.parse_args()
        emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
        landmarkDetector = LandmarkDetector()

        if args.task == 'predict':
            try:
                datasetPredictor = DatasetPredictor(emotions,'models/emotion_evaluation_1.1_model.sav',landmarkDetector)
                datasetPredictor.Predict()

            except (KeyboardInterrupt, SystemExit):
                datasetPredictor.Stop()
        elif args.task == 'train':
            datasetTrainer = DatasetTrainer(emotions, landmarkDetector)
            datasetTrainer.Train()
        elif args.task == 'prep':
            prep = DatasetPrep(emotions)
            prep.Run()
        else:
            print('Invalid arguments use (predict/train/prep)!')

if __name__ == "__main__":
    App()
