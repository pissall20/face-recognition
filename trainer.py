# -*- coding: utf-8 -*-
"""
Created on Wed May 24 07:47:27 2017

@author: Siddhesh
"""

import os
import cv2
import numpy as np
from PIL import Image

model = cv2.createLBPHFaceRecognizer()
path = '/home/hduser/Downloads/New folder/face_dataset/face_dataset'

def getid(path):
    imgpaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    ids = []
    for imgpath in imgpaths:
        faceimg = Image.open(imgpath).convert('L')
        facenp = np.array(faceimg,'uint8')
        ID = int(os.path.split(imgpath)[-1].split(".")[1])
        faces.append(facenp)
        ids.append(ID)
        cv2.imshow("training",facenp)
        cv2.waitKey(10)
    return np.array(ids), faces
ids, faces = getid(path)
model.train(faces,ids)
model.save("traindata.yml")
cv2.destroyAllWindows()
        
    

    
