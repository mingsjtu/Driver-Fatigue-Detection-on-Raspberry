# -*- coding: utf-8 -*-
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import RPi.GPIO as IO
import blinkBuzz as bb
import BLEthread as bt
import threading

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
def mouth_aspect_ratio(mouth,xmouth):
	A = distance.euclidean(mouth[3], mouth[9])
	B = distance.euclidean(xmouth[2], xmouth[6])
	
	a = B/A
	return a

thresh_1=0.4
thresh_2=0.6

thresh = 0.25
frame_check = 20
#使用dlib自带的frontal_face_detector作为我们的人脸提取器
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# detect = dlib.get_frontal_face_detector()
#用官方提供的模型构建特征提取器
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
mStart=48
mEnd=60
xStart=60
xEnd=68
flag_1=0;
flag_2=0;
drowsy=0;
priva=5
#priva = input("Please input your sleepy intend: (1-5,and 5 means the most) ")
thresh -= priva/5*0.05

# start the video stream thread
print("[INFO] camera sensor warming up...")
# vs = cv2.VideoCapture(0)
# vs = WebcamVideoStream(src=0).start()
cap = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

#cap=cv2.VideoCapture(0)
flag=0
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)
		mouth = shape[mStart:mEnd]
		xmouth = shape[xStart:xEnd]
		moutha = mouth_aspect_ratio(mouth,xmouth)

		print(mouth_aspect_ratio(mouth,xmouth))
		mouth = cv2.convexHull(mouth)
		xmouth = cv2.convexHull(xmouth)
		cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [xmouth], -1, (0, 255, 0), 1)

		if moutha > thresh_1:
		    flag_1 += 1
		    print (flag_1)
		    if flag_1 >= frame_check:
		        drowsy +=1
		        flag_1 =0
		        
		    if moutha >thresh_2:
		        flag_2 +=1
		        print(flag_2)
		        if flag_2 >=frame_check:
		            drowsy=2
		            flag_2=0
		    if drowsy>=2:
		        cv2.putText(frame, "****************ALERT!****************", (10, 30),
		        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		        cv2.putText(frame, "****************ALERT!****************", (10,325),
		        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("Drowsy")
		else:
		    drowsy = 0

		leftEye = shape[lStart:lEnd] 
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("Drowsy")
		else:
			flag = 0

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.stop()
