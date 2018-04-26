'''
Prepare training, validation, test data generators
'''
import numpy as np
import keras
import glob
import h5py
import multiprocessing
from helpers import labels_to_text

class LipNetDataGen(keras.callbacks.Callback):
    def __init__(self, path, batch_size, img_c, img_w, img_h, frames_n, absolute_max_string_len=32, shuffle=True):

        self.batch_size = batch_size
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.shuffle = shuffle

        self.files = glob.glob(path)
        self.indexes = np.arange(len(self.files))
        self.curr_index = multiprocessing.Value('i', -1)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __next__(self):

        with self.curr_index.get_lock():
            if (self.curr_index.value+2)*self.batch_size >= len(self.files):
                self.curr_index.value = 0
                np.random.shuffle(self.indexes)
            else:
                self.curr_index.value += 1

        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[self.curr_index.value * self.batch_size:(self.curr_index.value + 1) * self.batch_size]

        # Find list of IDs
        files_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(files_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, files_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.frames_n, self.img_w, self.img_h, self.img_c))
        y = np.empty((self.batch_size, self.absolute_max_string_len), dtype=int)
        label_length = [self.frames_n]*self.batch_size
        input_length = [self.absolute_max_string_len]*self.batch_size
        source_str = []

        # Generate data
        for i, ID in enumerate(files_temp):
            # Store sample
            input = h5py.File(ID, 'r')
            x_temp = np.asarray(input['video'].value)
            x_temp = np.transpose(x_temp, (2, 0, 1))
            x_temp = np.expand_dims(x_temp, axis=3)
            x_temp[np.isneginf(x_temp)] = 0.0
            x_temp[np.isinf(x_temp)] = 1.0
            x_temp[np.isnan(x_temp)] = 0.0
            x_temp[x_temp > 1.0] = 1.0
            x_temp[x_temp < 0.0] = 0.0            
#            print (np.isfinite(x_temp).all(), np.amax(x_temp), np.amin(x_temp))

            y_temp = input['label'].value
            y_temp = np.asarray(y_temp)
            y_temp[y_temp == -1] = 27

            X[i,] = x_temp
            y[i,] = y_temp
            if self.absolute_max_string_len  == 1:
                source_str.append(labels_to_text([y_temp.astype(int).tolist()]))
            else:
                source_str.append(labels_to_text(y_temp.astype(int).tolist()))
            
        label_length = np.asarray(label_length)
        input_length = np.asarray(input_length)
        source_str = np.asarray(source_str)

        inputs = {'input': X,
                  'labels': y,
                  'input_len': input_length,
                  'label_len': label_length,
                  'source_str': source_str
                  }
        outputs = {'ctc': np.zeros([self.batch_size])}  # dummy data for dummy loss function

        return (inputs, outputs)
