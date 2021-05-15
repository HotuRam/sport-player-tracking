# single-object-tracking-using your web ca-using-opencv-in-python
# object detection using imutils 


import cv2
import imutils
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
###############################################################################################
TrDict = {'csrt': cv2.legacy.TrackerCSRT_create,
          'kcf' : cv2.TrackerKCF_create,
        #   'boosting' : cv2.TrackerBoosting_create,
          'mil' : cv2.TrackerMIL_create,
        #   'tid' : cv2.TrackerTLD_create,
        #   'medianflow' : cv2.TrackerMedianFlow_create,
        #   'mosse' : cv2.TrackerMOSSE_crate
          }



tracker = TrDict['csrt']()
#tracker = cv2.TrackerCSRT_create()


# v = cv2.VideoCapture(r'C:\Users\Hotu Ram\OneDrive\Desktop\python\notebook\mot.mp4') # video
v = cv2.VideoCapture(0)


ret, frame = v.read()
frame = imutils.resize(frame,width=800) # you can resize the frame of video
cv2.imshow('Frame',frame)
bb = cv2.selectROI('Frame',frame)
tracker.init(frame,bb)


while True:
    ret, frame = v.read()
    if not ret:
        break
    frame = imutils.resize(frame,width=800)  # you can resize the frame of video
    (success,box) = tracker.update(frame)
    if success:
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
v.release()
cv2.destroyAllWindows()

