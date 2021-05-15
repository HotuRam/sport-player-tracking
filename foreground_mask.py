import cv2
import numpy as np
path =r'videos\match.mp4'

cap = cv2.VideoCapture(path)
d={'gmg':cv2.bgsegm.createBackgroundSubtractorGMG() ,
'mog2':cv2.createBackgroundSubtractorMOG2(detectShadows=False) ,
'knn':cv2.createBackgroundSubtractorKNN(detectShadows=True) ,
'mog':cv2.bgsegm.createBackgroundSubtractorMOG() 
}
print('running ...')
fgbg = d['gmg']

while True:
    success, frame = cap.read()
    if frame is None: 
        break
    fgmask = fgbg.apply(frame)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG MASK Frame', fgmask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
cap.release()
cv2.destroyAllWindows()