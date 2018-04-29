import numpy as np
import cv2
import h5py


def get_segment(start, end, video):
    ind = 0
    segment = np.empty([112,112,end-start+1])
    cap = cv2.VideoCapture(video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            ind +=1
            if ind>=start and ind<=end:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, dsize=(112, 112))
                normalized_image = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                segment[:,:,ind-start] = normalized_image
        else:
            break
    cap.release()
    return segment

def create(input_filename, output_filename):
    video = get_segment(1, 75, input_filename)
    out = h5py.File(output_filename, 'w')
    out.create_dataset('video', data=video)
    out.close()

create('AFTERNOON.mp4','AFTERNOON.hdf5')