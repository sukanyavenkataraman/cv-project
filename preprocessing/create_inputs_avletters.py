# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 18:54:36 2018

@author: pragathi
"""

import glob
import scipy.io as sio
import numpy as np
import h5py
import os
import cv2

def get_max_length():
    frames = []
    for file in glob.glob('../data/avletters/avletters/Lips/*.mat'):
        sample = sio.loadmat(file)
        sh = sample['siz'][0]
        frames.append(sh[2])

    frames = np.asarray(frames)
    return max(frames)
	
max_length = get_max_length()

if not os.path.exists('../data/avletters_processed'):
    os.makedirs('../data/avletters_processed')

for f, file in enumerate(glob.glob('../data/avletters/avletters/Lips/*.mat')):
    # load mat file
    sample = sio.loadmat(file)
    sh = sample['siz'][0]
    sh = sh.astype(int)
    sample = np.reshape(sample['vid'],(sh[1],sh[0],sh[2]))
    sample = sample.transpose(1, 0, 2)
    # resize the video to 112x112
    resized_sample = np.zeros((112,112,sh[2]))
    for i in range(sh[2]):
	resized_sample[:,:,i] = cv2.resize(sample[:,:,i], dsize=(112, 112))
	resized_sample[:,:,i] = cv2.normalize(resized_sample[:,:,i], None, 0, 255, cv2.NORM_MINMAX) 
    
    # make all vides the same number of frames
    padding_length = max_length - sh[2]
    front_padding = np.zeros((112,112,int(np.ceil(padding_length/2))))
    rear_padding = np.zeros((112,112,int(np.floor(padding_length/2))))
    resized_sample = np.concatenate((front_padding, resized_sample, rear_padding), axis=2)
    
    # save the file as hdf5 dict
    label = file.split('Lips')[1][1]
    label = ord(label)-ord('A')
    out = h5py.File('../data/avletters_processed/{i}.hdf5'.format(i=str(f+1)),'w')
    out.create_dataset('video', data=resized_sample)
    out.create_dataset('label', data=label)
    out.close()
