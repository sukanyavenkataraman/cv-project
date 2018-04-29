import crop
import cv2
import numpy as np
import h5py

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def sparse_optical_flow(filename):
    input_ = h5py.File(filename,'r')
    video = np.asarray(input_['video'].value).astype(np.uint8)
    prev = video[:,:,0]

    region = 'lipsjaws'
    pts_prev = crop.landmarks(prev,region)
    if region is 'lipsjaws':
        flow_feature = np.empty([37, video.shape[2] - 1])
    if region is 'lips':
        flow_feature = np.empty([20, video.shape[2] - 1])

    pts_prev = np.array([[pts] for pts in pts_prev]).astype(np.float32)
    color = np.random.randint(0,255,[100,3])
    mask = np.zeros_like(prev)
    for i in range(1,video.shape[2]):
        curr = video[:,:,i]
        if sum(curr.flatten())==0:
            continue
        pts_curr, st, err = cv2.calcOpticalFlowPyrLK(prev, curr, pts_prev, None, **lk_params)

        """
        for j, (new, old) in enumerate(zip(pts_curr, pts_prev)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[j].tolist(), 2)
            frame = np.asarray(curr).astype(np.uint8)
            frame = cv2.circle(frame, (a, b), 5, color[j].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey() & 0xff
        if k == 27:
            break
        """
        flow = pts_curr - pts_prev
        A = np.asarray([f[0][0] for f in flow])
        B = np.asarray([f[0][1] for f in flow])
        flow_feature[:,i-1] = np.sqrt(np.square(A) + np.square(B))
        prev = curr
        pts_prev = pts_curr

    return flow_feature


def dense_optical_flow(filename):
    input_ = h5py.File(filename,'r')
    video = np.asarray(input_['video'].value).astype(np.uint8)
    prev = video[:,:,0]
    flow_feature = np.empty([112, 112, video.shape[2]-1])
    for i in range(1,video.shape[2]):
        curr = video[:,:,i]
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_feature[:, :, i-1] = np.sqrt(np.square(flow[...,0]) + np.square(flow[...,0]))
        prev = curr

    return flow_feature

#sparse_optical_flow('AFTERNOON.hdf5')
#dense_optical_flow('1.hdf5')