import glob
from shutil import copyfile
import cv2

class DatasetPrep:
    def __init__(self, classList, landmarkDetector):
        self.classList = classList

        self.CleanDataset()
        self.CollectFaces()

    # TODO move this to a different class
    def CleanDataset(self):
        participants = glob.glob('dataset/source_emotion/*', recursive=True)
        for x in participants:
            part = '%s' %x[-4:]

            for sessions in glob.glob('%s/*' %x): 
                for files in glob.glob('%s/*' %sessions):
                    current_session = files[28:-30]
                    file = open(files, 'r')
                    emotion = int(float(file.readline()))

                    if(len(self.classList) - 1 < emotion):
                        continue

                    source_files = sorted(glob.glob('dataset/source_image/%s/%s/*' %(part, current_session)))

                    source_emotion = source_files[-1]
                    source_neutral = source_files[0]

                    copyfile(source_neutral, 'dataset/sorted_set/%s/%s' %('neutral', source_neutral[30:]))
                    copyfile(source_emotion, 'dataset/sorted_set/%s/%s' %(self.classList[emotion], source_emotion[30:]))
                    print('copied')
        

    # TODO move this to a different class
    def CollectFaces(self):
        face_detect = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

        for emotion in self.classList:
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