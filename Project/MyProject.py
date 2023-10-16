import ArucoModule as arm
import cv2
import cv2.aruco as aruco
import numpy as np
import os




cap = cv2.VideoCapture(0)
augDics = arm.loadAugImages("Markers")
while True:
    success, img = cap.read()
    arucoFound = arm.findArucoMarkers(img)

    #Loop through all the markers and augment each one
    if len(arucoFound[0])!=0:
        for bbox, id in zip(arucoFound[0],arucoFound[1]):
            if int(id) in augDics.keys():
                img = arm.augmentAruco(bbox,id,img,augDics[int(id)])

    cv2.imshow("Image",img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break