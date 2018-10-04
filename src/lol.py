import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier("data/cascades/haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rect = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)

    for (x, y, w, h) in rect:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('frames', frame)
    if cv2.waitKey(20) and 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()