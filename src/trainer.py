import sys
from EmotionTrainer import EmotionTrainer

class Trainer:
    def __init__(self):
        try:
            self.emotionTrainer = EmotionTrainer()
            self.emotionTrainer.Train()

            self.emotionTrainer.Predict()

        except (KeyboardInterrupt, SystemExit):
            self.Stop()

    def Stop(self):
        self.emotionTrainer.Stop()

if __name__ == "__main__":
    Trainer()
