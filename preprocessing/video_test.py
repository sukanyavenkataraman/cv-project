# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import crop
import cv2

video_file = 'AFTERNOON.mp4'
cap = cv2.VideoCapture(video_file)

while(cap.isOpened()): 
    ret, frame = cap.read()
    if ret == True:
        I = crop.detect_mouth(frame)
        cv2.imshow('frame', I)
        cv2.waitKey()
    else:
        break
     
cap.release()
cv2.destroyAllWindows()
