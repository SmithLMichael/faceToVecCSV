# This file contains the implementation to extract facial features from an image
#
# Author: Michael L. Smith
# 
# Many methods and implementation taken from:
# https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ 

from imutils import face_utils
import numpy as np
import argparse
import imutils
import cv2
import dlib
import csv

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords


def draw_bounding_box_and_points(image, rect, face_num, shape):

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	 
		# show the face number
		cv2.putText(image, "Face #{}".format(face_num + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	 
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


def main(args):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(args["image"])
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# there will only ever be one face in the image, etc

	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		for row in shape:
			print str(row[0]) + ', ' + str(row[1])
	 
	 	# uncomment to draw box and points
		# draw_bounding_box_and_points(image, rect, i, shape)

	# uncomment to show image
	# show the output image with the face detections + facial landmarks
	# cv2.imshow("Output", image)
	# cv2.waitKey(0)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", required=True,
		help="path to facial landmark predictor")
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	args = vars(ap.parse_args())

	main(args)




