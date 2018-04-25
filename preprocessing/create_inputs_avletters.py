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
	sample = sio.loadmat(file)
	sh = sample['siz'][0]
	sh = sh.astype(int)
	sample = np.reshape(sample['vid'],(sh[1],sh[0],sh[2]))
	sample = sample.transpose(1, 0, 2)
	padding_length = max_length - sh[2]
	front_padding = np.zeros((sh[0],sh[1],int(np.ceil(padding_length/2))))
	rear_padding = np.zeros((sh[0],sh[1],int(np.floor(padding_length/2))))
	sample = np.concatenate((front_padding, sample,rear_padding), axis=2)
	
	label = file.split('Lips')[1][1]
	label = ord(label)-ord('A')
	out = h5py.File('../data/avletters_processed/{i}.hdf5'.format(i=str(f+1)),'w')
	out.create_dataset('video', data=sample)
	out.create_dataset('label', data=label)
	out.close()
