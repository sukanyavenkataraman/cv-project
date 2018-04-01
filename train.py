'''
Trains the model
'''

from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras import metrics

import numpy as np
from model import LipNetModel
import h5py
import glob

np.random.seed(42)

def train(path, epochs=10, img_c=1, img_w=112, img_h=112, frames_n=75, absolute_max_string_len=32, batch_size=32, learning_rate=0.0001, output_dir='saved_models'):
    print ('Starting training...')

    files = glob.glob(path)
    print (files)
    data = []
    labels = []


    for file in files:
        input = h5py.File(file, 'r')

        data.append(input['video'].value)
        labels.append(input['label'].value)

    data = np.asarray(data)
    labels = np.asarray(labels)
    print(data.shape, labels.shape)

    data = np.transpose(data, (0, 3, 2, 1))
    data = np.expand_dims(data, axis=4)

    print(data.shape, labels.shape)
    input_len = [75]*len(data)
    output_len = [32]*len(data)

    train_model = LipNetModel(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n, absolute_max_string_len=absolute_max_string_len, output_size=28)

    # Dummy function since the model calculates the loss
    train_model.model.compile(loss={'ctc': lambda y_true, y_pred:y_pred}, optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=[metrics.categorical_accuracy])

    checkpoint = ModelCheckpoint(output_dir+'/'+"lipnet{epoch:02d}.h5", monitor='val_loss', verbose=1, mode='auto', period=1)

    train_model.model.fit({'input':np.array(data), 'labels':np.array(labels), 'input_len':np.array(input_len), 'label_len':np.array(output_len)}, {'ctc':np.zeros([batch_size])}, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_split=0.0)

train('/Users/sukanya/Downloads/data/*.hdf5', batch_size=2)