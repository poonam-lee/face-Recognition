import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier("C:/Users/hp/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
img = cv2.imread("C:/Users/hp/Desktop/task1.jpg",1)
#print(img)
gray_img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors= 5)
#print(type(faces))
#print(faces)
for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,225,0),3)
resized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
cv2.imshow("gray",resized)
cv2.waitKey(0)
