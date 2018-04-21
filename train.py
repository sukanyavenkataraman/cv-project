'''
Trains the model
'''

from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras import metrics

import numpy as np
from model import LipNetModel
import h5py
import glob
import random
import os

np.random.seed(22)
from keras import backend as K

K.set_learning_phase(1)

os.environ["CUDA_VISIBLE_DEVICES"]="3"
def train(path, epochs=10, img_c=1, img_w=112, img_h=112, frames_n=75, absolute_max_string_len=32, batch_size=32, learning_rate=0.1, output_dir='saved_models'):
    print ('Starting training...')

    files = glob.glob(path)
    #files = files[:batch_size+2]
    random.shuffle(files)	
	
    print (files)
    data = []
    labels = []


    for file in files:
        input = h5py.File(file, 'r')
        data.append(input['video'].value)
        labels.append(input['label'].value)

    data = np.asarray(data)
    labels = np.asarray(labels)
    labels[labels == -1] = 27
    print(data.shape, labels.shape)

    data = np.transpose(data, (0, 3, 2, 1))
    data = np.expand_dims(data, axis=4)

    print(data.shape, labels.shape)
    input_len = [73]*len(data)
    output_len = [32]*len(data)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    train_model = LipNetModel(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n, absolute_max_string_len=absolute_max_string_len, output_size=28)

    # Dummy function since the model calculates the loss
    #train_model.model.compile(loss={'ctc': lambda y_true, y_pred:y_pred}, optimizer=SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=200), metrics=[metrics.categorical_accuracy])

    train_model.model.compile(loss={'ctc': lambda y_true, y_pred:y_pred}, optimizer=adam, metrics=[metrics.categorical_accuracy])
    checkpoint = ModelCheckpoint(output_dir+'/'+"lipnet{epoch:02d}.h5", monitor='val_loss', verbose=0, mode='auto', period=1)

    train_model.model.fit({'input':np.array(data[:-2]), 'labels':np.array(labels[:-2]), 'input_len':np.array(input_len[:-2]), 'label_len':np.array(output_len[:-2])}, {'ctc':np.zeros([len(data[:-2])])}, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_split=0.0)
	
    K.set_learning_phase(0)
    print(train_model.model.evaluate({'input':np.array(data[-2:]), 'labels':np.array(labels[-2:]), 'input_len':np.array(input_len[-2:]), 'label_len':np.array(output_len[-2:])}, {'ctc':np.zeros([2])}))

train('data/*.hdf5', batch_size=20, epochs=20)
