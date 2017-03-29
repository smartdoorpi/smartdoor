#code to detect faces
#python
#opencv

import numpy as np
import cv2
import os
os.chdir('F:\Shubham\My files\Embedded Class\OpenCV-Python\CWD')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while(1):
	ret,img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	equ = cv2.equalizeHist(gray)
	faces = face_cascade.detectMultiScale(equ,1.25,4)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			
	cv2.imshow('Faces',img)
	cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
