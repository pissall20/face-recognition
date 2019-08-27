# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:15:52 2017

@author: Siddhesh
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sqlite3

samplenum = 0
cam = cv2.VideoCapture(0)
recog = cv2.createLBPHFaceRecognizer()
recog.load("home/hduser/Downloads/New Folder/face_dataset/traindata.yml")
facedetect = cv2.CascadeClassifier(
    "home/hduser/Downloads/New Folder/facedetection/haarcascade_frontalface_default.xml"
)


def getID(ID):
    connection = sqlite3.connect(
        "home/hduser/Downloads/New Folder/face_dataset/database.db"
    )
    cmd = "SELECT * FROM staff where ID=" + str(ID)
    details = connection.execute(cmd)
    profile = None
    for det in details:
        profile = det
    connection.close()
    return profile


# id = raw_input("enter the ID:  ")
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_PLAIN, 2, 1, 0, 2)
while cam.isOpened():  # check !
    # capture frame-by-frame
    ret, img = cam.read()

    if ret:  # check ! (some webcam's need a "warmup")
        # our operation on frame come here
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 3)
            ID, conf = recog.predict(gray[y : y + h, x : x + w])
            profile = getID(ID)
            if profile != None:
                cv2.cv.PutText(
                    cv2.cv.fromarray(img), str(profile[1]), (x, y + h + 50), font, 255
                )
                cv2.cv.PutText(
                    cv2.cv.fromarray(img), str(profile[2]), (x, y + h + 70), font, 255
                )
                cv2.cv.PutText(
                    cv2.cv.fromarray(img), str(profile[3]), (x, y + h + 90), font, 255
                )

        cv2.imshow("frame", img)  # Display the resulting frame
        if cv2.waitKey(1) == ord("q"):
            break


#
cam.release()
cv2.destroyAllWindows()
