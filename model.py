from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.core import Lambda
from keras import backend as K
import numpy as np
from resnet import ResnetBuilder

# CTC Layer implementation using Lambda layer
# (because Keras doesn't support extra prams on loss function)
def CTC(name, args):
    return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)

# Actual loss calculation
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # From Keras example image_ocr.py:
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :]
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

class LipNetModel(object):
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=75, absolute_max_string_len=32, output_size=28, onlyRNN = False):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.output_size = output_size
        self.onlyRNN = onlyRNN

        self.buildModel()

    def buildModel(self):

        if K.image_data_format() == 'channels_first':
            input_shape = (self.img_c, self.frames_n, self.img_w, self.img_h)
        else:
            input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)

        self.input_data = Input(name='input', shape=input_shape, dtype=np.float32)
        self.labels = Input(name='labels', shape=[self.absolute_max_string_len], dtype=np.float32)
        self.input_length = Input(name='input_len', shape=[1], dtype=np.int64)
        self.label_length = Input(name='label_len', shape=[1], dtype=np.int64)

        if not self.onlyRNN:
            pad = ZeroPadding3D(padding=(1, 2, 2), name='pad')(self.input_data)
            conv1 = Conv3D(64, (5, 7, 7), strides=(1, 2, 2), data_format='channels_last', activation='relu', kernel_initializer='he_normal', name='conv1')(pad)
            maxpool = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), name='max1')(conv1)

            print (maxpool.shape)

            # TODO: check if hardcoded 256 can be removed
            resnet = TimeDistributed(ResnetBuilder.build_resnet_34((int(maxpool.shape[-1]), int(maxpool.shape[-2]), int(maxpool.shape[-3])), 256), name='timedistresnet')(maxpool)

        else:
            resnet = self.input_data

        gru1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(resnet)
        gru2 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(gru1)

        dense = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(gru2)

        self.y_pred = Activation('softmax', name='softmax')(dense)

        self.loss = CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

        self.model = Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length], outputs=self.loss)

        print (self.model.summary())

    def predict(self, input_batch):
        return self.test_function([input_batch])[0]  # the first 0 indicates test

    @property
    def test_function(self):
        # captures output of softmax so we can decode the output during visualization
        #print (self.input_data, K.learning_phase(), self.y_pred, K.learning_phase())
        return K.function([self.input_data], [self.y_pred])