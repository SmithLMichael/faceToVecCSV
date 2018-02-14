# faceToVecCSV
Converts faces into vectorized datapoints, stored in CSV format

# Many methods and implementation taken from:
https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

# ^^ you can also use this link to understand how to use the python script - or download dlib or opencv through the links on the page

How to use currently:

you need opencv and dlib installed

You need to download a file shape_predictor_68_face_landmarks.dat from:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

then you invoke this program on the terminal using:

python faceToCSV.py --shape-predictor shape_predictor_68_face_landmarks.dat --image frame_0.jpg > frame0.csv
