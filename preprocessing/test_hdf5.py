import h5py
import numpy as np
import cv2

filename = '1.hdf5'
input_ = h5py.File(filename, 'r')
video = np.asarray(input_['video'].value).astype(np.uint8)
for i in range(0, video.shape[2]):
    curr = video[:, :, i]
    cv2.imshow('frame', curr)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()