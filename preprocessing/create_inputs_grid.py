# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:57:05 2018

@author: pragathi
"""
import numpy as np
import cv2
import crop
import h5py
import zipfile, tarfile
import os

def get_segment(start, end, video):
    ind = 0
    segment = np.empty([112,112,end-start+1])
    cap = cv2.VideoCapture(video)
    while(cap.isOpened()): 
        ret, frame = cap.read()
        if ret == True:
            ind +=1
            if ind>=start and ind<=end:
                segment[:,:,ind-start] = crop.detect_mouth(frame)
        else:
            break 
    cap.release()
    return segment

def get_sentence_label(label_file, N):
    label = []
    for line in label_file:     
        if ('sil' or 'sp') not in line:
            word = [ord(ch)-ord('a') for ch in line.split()[2]]
            if len(label):
                label = label + [26] + word
            else:
                    label = word
    
    padding = np.ones((N-len(label))) * 27
    return np.concatenate((np.array(label), padding), axis=0)

if not os.path.exists('../data/grid_processed'):
    os.makedirs('../data/grid_processed')

counter = 0
##### change this range for the final training #####
speaker_begin = 1
speaker_end = 1
for i in range(speaker_begin,speaker_end+1):
    video_file = '../data/grid/s{num}.mpg_vcd.zip'.format(num=str(i))
    video_dir = zipfile.ZipFile(video_file, 'r')
    label_file = '../data/grid/s{num}.tar'.format(num=str(i))
    label_dir = tarfile.open(label_file, 'r')
    label_dir.extractall()
    
    total = len(video_dir.namelist())
    data = []
    labels = []
    sentences = []
    ##### change this range for the final training #####
    for video_name in video_dir.namelist():     
        if '.mpg' in video_name:
            sentence = video_name.split('/')[1].split('.')[0]
            
            counter +=1
            print('{} to go ...'.format(total - counter - 1))
            print('working on {} ...'.format(sentence))
            
            for label_name in label_dir.getnames(): 
                if sentence in label_name:
                    label_file    = open(label_name,'r') 
            
            label = get_sentence_label(label_file, 32)     
            video = get_segment(1, 75, video_name)            
        
            out = h5py.File('../data/grid_processed/{i}-{sentence}.hdf5'.format(i=str(i),sentence=sentence),'w')
            out.create_dataset('video', data=video)
            out.create_dataset('label', data=labels)
            out.close()
  
            
print('...done') 