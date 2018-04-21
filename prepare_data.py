'''
Prepare training, validation, test data generators
'''
import numpy as np
import keras
import glob
import h5py

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

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        files_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(files_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, files_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.frames_n, self.img_h, self.img_w, self.img_c))
        y = np.empty((self.batch_size, self.absolute_max_string_len), dtype=int)
        label_length = [self.frames_n]*self.batch_size
        input_length = [self.absolute_max_string_len]*self.batch_size

        # Generate data
        for i, ID in enumerate(files_temp):
            # Store sample
            input = h5py.File(ID, 'r')
            x_temp = np.asarray(input['video'].value)
            x_temp = np.transpose(x_temp, (2, 1, 0)) # TODO: /255?!
            x_temp = np.expand_dims(x_temp, axis=3)

            y_temp = np.asarray(input['label'].value)
            y_temp = np.asarray(y_temp)
            y_temp[y_temp == -1] = 27

            X[i,] = x_temp
            y[i,] = y_temp

        label_length = np.asarray(label_length)
        input_length = np.asarray(input_length)

        inputs = {'the_input': X,
                  'the_labels': y,
                  'input_length': input_length,
                  'label_length': label_length
                  }
        outputs = {'ctc': np.zeros([self.batch_size])}  # dummy data for dummy loss function

        return (inputs, outputs)