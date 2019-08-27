# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:59:51 2017

@author: Siddhesh
"""

import cv2
import numpy

samplenum = 0
facedetect = cv2.CascadeClassifier("facedetection/haarcascade_frontalface_default.xml")

di = raw_input("enter an ID :")
cam = cv2.VideoCapture(0)

ret, img = cam.read()
if ret:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        samplenum = samplenum + 1
        cv2.imwrite(
            "face_dataset/user" + "." + str(id) + "." + str(samplenum) + ".jpg",
            gray[y : y + h, x : x + w],
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("face", img)
    cv2.waitKey(1)
    if samplenum > 20:
        cam.release()
        cv2.destroyAllWindows()
