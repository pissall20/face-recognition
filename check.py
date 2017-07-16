# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:05:25 2017

@author: Siddhesh
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sqlite3

ID = raw_input("enter the ID:  ")
Name = raw_input("enter the name : ")
samplenum = 0
cam = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('home/hduser/Downloads/New Folder/facedetection/haarcascade_frontalface_default.xml')

def update(ID, Name) :
    connection = sqlite3.connect("home/hduser/Downloads/New Folder/face_dataset/database.db")
    cmd = "SELECT * from staff where ID="+str(ID)
    details = connection.execute(cmd)
    recordexist = 0
    for row in details:
        recordexist = 1
    if(recordexist==1):
        cmd = "UPDATE staff set Name="+str(Name)+" where ID="+str(ID)
    else:
        cmd = "INSERT into staff(ID,Name) values("+str(ID)+","+str(Name)+")"
        connection.execute(cmd)
        connection.commit()
        connection.close()
            
update(ID,Name)

while(cam.isOpened()):  # check !
    # capture frame-by-frame
    ret, img = cam.read()

    if ret: # check ! (some webcam's need a "warmup")
        # our operation on frame comes here
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(img,1.3,5)
        for(x,y,w,h) in faces :
            samplenum = samplenum+1
            cv2.rectangle(img,(x,y), (x + w, y + h),(0,0,255),2)
            cv2.imwrite("face_dataset/face_dataset/user"+"." + str(ID)+ "." +str(samplenum)+".jpg",gray[y:y+h,x:x+w])
            
            
        cv2.imshow('frame', img) # Display the resulting frame
        cv2.waitKey(1)
        if(samplenum>50):
            #if(cv2.waitKey(1)==ord('q')):
            break;
        
    
# When everything is done release the capture
cam.release()
cv2.destroyAllWindows()
