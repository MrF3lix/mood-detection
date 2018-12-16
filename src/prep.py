import sys
from DatasetPrep import DatasetPrep

class Prep:
    def __init__(self):
        emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
        prep = DatasetPrep(emotions)
        prep.Run()

if __name__ == "__main__":
    Prep()
