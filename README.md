# Mood detection

A small application to classify the mood of one or more faces captured by a webcam. 

## Setup

To run this application locally you need to have the following things installed.

- Python 3 > 
- OpenCV
- Dlib
- Sklearn
- Pickel

The dataset I used is Cohn-Kanade (CK and CK+) database.
http://www.consortium.ri.cmu.edu/ckagree/

Once the dataset is downloaded run `python3 prep.py` to cleanup the dataset and sort it correctly.

After sorting and cleaning up the model needs to be trained. This can be done by running `python3 trainer.py`.

And to use the mood detection after creating the model run `python3 app.py`.

## Further improvements
- The emotions that should be detected are now defined in 3 sperate files, this should be unified.
- The accuracy can be improved by using more data and making sure that each data class has a similar amount of training data.
