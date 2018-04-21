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

np.random.seed(22)

def train(path, epochs=10, img_c=1, img_w=112, img_h=112, frames_n=75, absolute_max_string_len=32, batch_size=32, learning_rate=0.0001, output_dir='saved_models'):
    print ('Starting training...')

    files = glob.glob(path)
    files = files[:batch_size+2]
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

    train_model = LipNetModel(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n, absolute_max_string_len=absolute_max_string_len, output_size=28)

    # Dummy function since the model calculates the loss
    train_model.model.compile(loss={'ctc': lambda y_true, y_pred:y_pred}, optimizer=SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=200), metrics=[metrics.categorical_accuracy])

    checkpoint = ModelCheckpoint(output_dir+'/'+"lipnet{epoch:02d}.h5", monitor='val_loss', verbose=0, mode='auto', period=1)

    train_model.model.fit({'input':np.array(data[:batch_size]), 'labels':np.array(labels[:batch_size]), 'input_len':np.array(input_len[:batch_size]), 'label_len':np.array(output_len[:batch_size])}, {'ctc':np.zeros([batch_size])}, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_split=0.0)
	
    print(train_model.model.evaluate({'input':np.array(data[-2:]), 'labels':np.array(labels[-2:]), 'input_len':np.array(input_len[-2:]), 'label_len':np.array(output_len[-2:])}, {'ctc':np.zeros([2])}))

train('dataset/*.hdf5', batch_size=3, epochs=5)