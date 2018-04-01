'''
Trains the model
'''

from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras import metrics

import numpy as np
from model import LipNetModel
import h5py

np.random.seed(42)

def train(filename, epochs=10, img_c=1, img_w=112, img_h=112, frames_n=75, absolute_max_string_len=32, batch_size=32, learning_rate=0.0001, output_dir='saved_models'):
    print ('Starting training...')
    all_data = h5py.File(filename, 'r')

    data = all_data['video'].value
    labels = all_data['label'].value

    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=3)
    data = np.expand_dims(data, axis=0)
    labels = np.expand_dims(labels, axis=0)

    print(data.shape, labels.shape, list(labels))
    train_model = LipNetModel(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n, absolute_max_string_len=absolute_max_string_len, output_size=28)

    # Dummy function since the model calculates the loss
    train_model.model.compile(loss={'ctc': lambda y_true, y_pred:y_pred}, optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=[metrics.categorical_accuracy])

    checkpoint = ModelCheckpoint(output_dir+'/'+"lipnet{epoch:02d}.h5", monitor='val_loss', verbose=1, mode='auto', period=1)

    train_model.model.fit({'input':np.array(data), 'labels':np.array(labels), 'input_len':np.array([75]), 'label_len':np.array([32])}, {'ctc':np.zeros([1])}, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_split=0.0)

train('/Users/sukanya/Downloads/1.hdf5')