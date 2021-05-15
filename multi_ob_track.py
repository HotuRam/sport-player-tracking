import cv2
import imutils
import numpy as np
TrDict = {'csrt': cv2.legacy.TrackerCSRT_create,
         'kcf' : cv2.TrackerKCF_create,
        #  'boosting' : cv2.TrackerBoosting_create,
         'mil': cv2.TrackerMIL_create,
        #  'tld': cv2.TrackerTLD_create,
        #  'medianflow': cv2.TrackerMedianFlow_create,
        #  'mosse':cv2.TrackerMOSSE_create
         }
# trackers = cv2.legacy.MultiTracker_create()
trackers = cv2.legacy_MultiTracker.create() 
v = cv2.VideoCapture('videos/match.mp4')
ret, frame = v.read()
print(frame.shape)
k = 4
for i in range(k):
    frame = imutils.resize(frame,width=800)
    cv2.imshow('Frame',frame)
    
    bbi = cv2.selectROI('Frame',frame)
    tracker_i = TrDict['csrt']()
    trackers.add(tracker_i,frame,bbi)
frameNumber = 2
baseDir = r'./'

while True:
    ret, frame = v.read()
    if not ret:
        break
    (success,boxes) = trackers.update(frame)
    # np.savetxt(baseDir + '/frame_'+str(frameNumber)+'.txt',boxes,fmt='%f')
    frameNumber+=1
    for box in boxes:
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
v.release()
cv2.destroyAllWindows()
boxes = np.loadtxt('frame_102.txt')
print(boxes)
# array([[802.634583, 372.809296,  58.366409, 117.794022],
#        [887.393677, 239.489075,  40.519829,  84.808945],
#        [431.494141, 382.603577,  66.585564, 110.282341],
#        [951.417419, 254.674072,  56.862694,  88.235214]])
