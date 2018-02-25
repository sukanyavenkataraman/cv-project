# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:03:55 2018

@author: pragathi
"""

import numpy as np
import cv2
import dlib 

global detector, predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def bounding_points(coords):
    X = [x for (x,y) in coords]
    Y = [y for (x,y) in coords]
    return min(X), min(Y), max(X) - min(X), max(Y)- min(Y)

def detect_mouth(frame):
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # take the first face
    rect = detector(gray, 1)
    if rect:
        # get coordinates for the facial landmarks
        shape = predictor(gray, rect[0])
        # convert them to coordinates
        shape = shape_to_np(shape)
        # find bounding points of the mouth mask
        x,y,w,h = bounding_points(shape[48:68])
        # leave a margin
        margin = 0.2 
        dx = int(margin*w) 
        dy = int(margin*h) 
        # resize image 
        mouth_image = cv2.resize(gray[y-dy:y+h+dy, x-dx:x+w+dx], dsize=( 112, 112))
        normalized_image = cv2.normalize(mouth_image, None, 0, 255, cv2.NORM_MINMAX) 
    else:
        return np.empty([112,112])
    
    return normalized_image
        

