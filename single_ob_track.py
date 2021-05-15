# single-object-tracking-in-videos-using-opencv-in-python
# object detection using imutils 


import cv2
import imutils
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
###############################################################################################
TrDict = {'csrt': cv2.legacy.TrackerCSRT_create,
          'kcf' : cv2.legacy.TrackerKCF_create,
          'boosting' : cv2.legacy.TrackerBoosting_create,
          'mil' : cv2.legacy.TrackerMIL_create,
          'tid' : cv2.legacy.TrackerTLD_create,
          'medianflow' : cv2.legacy.TrackerMedianFlow_create,
        #   'mosse' : cv2.legacy.TrackerMOSSE_crate
          }



tracker = TrDict['csrt']()


v = cv2.VideoCapture('videos\match.mp4') # video


ret, frame = v.read()
frame = imutils.resize(frame,width=600)
cv2.imshow('Frame',frame)
bb = cv2.selectROI('Frame',frame)
tracker.init(frame,bb)


while True:
    ret, frame = v.read()
    if not ret:
        break
    frame = imutils.resize(frame,width=600)
    (success,box) = tracker.update(frame)
    if success:
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,0,0),2)
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
v.release()
cv2.destroyAllWindows()