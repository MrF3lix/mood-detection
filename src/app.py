import sys
import cv2
from FacialLandmarkTracking import FacialLandmarkTracking

class App:
    def __init__(self):
        try:

            self.landmarkTracking = FacialLandmarkTracking()
            while True:

                self.landmarkTracking.OutputWebcamFrame()
                k = cv2.waitKey(1)
                if k % 256 == 27:
                    self.Stop()
                    break

        except (KeyboardInterrupt, SystemExit):
            self.Stop()

    def Stop(self):
        self.landmarkTracking.Stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    App()
