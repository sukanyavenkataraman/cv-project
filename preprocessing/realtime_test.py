# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 19:08:24 2018

@author: pragathi
"""
"""
Run this file to open webcam and get lip segmented region of interest.
Resized to 112x112 and normalized.
"""
import crop
import cv2

cap = cv2.VideoCapture(0)
while(1): 
    ret, frame = cap.read()
    if ret == True:
        I = crop.detect_mouth(frame)
        cv2.imshow('frame', I)
        c = cv2.waitKey(1) 
        if c==27: #Esc key
            break
    else:
        break
     
cap.release()
cv2.destroyAllWindows()
