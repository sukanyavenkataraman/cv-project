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
from prepare_data import LipNetDataGen
K.set_learning_phase(1)
from helpers import labels_to_text
from helpers import Spell, Decoder
from callbacks import Statistics, Visualize
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="3"
PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY = 'grid.txt'

def train_GRID(train_path, valid_path, start_epoch=0, epochs=10, img_c=1, img_w=112, img_h=112, frames_n=75,
          absolute_max_string_len=32, batch_size=32, learning_rate=0.0001, output_dir='saved_models'):

    print ('Starting training...')
    train_num = len(glob.glob(train_path))
    valid_num  = len(glob.glob(valid_path))

    train_data = LipNetDataGen(train_path, batch_size, img_c, img_w, img_h, frames_n, absolute_max_string_len)
    valid_data = LipNetDataGen(valid_path, batch_size, img_c, img_w, img_h, frames_n, absolute_max_string_len)

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    train_model = LipNetModel(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n, absolute_max_string_len=absolute_max_string_len, output_size=28)

    # Dummy function since the model calculates the loss
    train_model.model.compile(loss={'ctc': lambda y_true, y_pred:y_pred}, optimizer=adam, metrics=[metrics.categorical_accuracy])

    # load weight if necessary
    if start_epoch > 0:
        weight_file = os.path.join(output_dir + '/', ('lipnet%02d.h5') % (start_epoch - 1))
        train_model.model.load_weights(weight_file)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                        postprocessors=[labels_to_text, spell.sentence])

    checkpoint = ModelCheckpoint(output_dir+'/'+"lipnet{epoch:02d}.h5", monitor='val_loss', verbose=0, mode='auto', period=50)

    # define callbacks
    statistics = Statistics(train_model, valid_data, decoder, 256, output_dir=os.path.join(output_dir+'/', 'stats'))
    visualize = Visualize(os.path.join(output_dir+'/'+'visualise'), train_model, valid_data, decoder,
                          num_display_sentences=batch_size)
    tensorboard = TensorBoard(log_dir=os.path.join(output_dir+'/'+'tensorboard'))
    csv_logger = CSVLogger(os.path.join(output_dir+'/'+'tensorboard', "{}.csv".format('training')), separator=',', append=True)

    steps_per_epoch_train = train_num/batch_size
    steps_per_epoch_valid = valid_num/batch_size
 
    train_model.model.fit_generator(generator=train_data,
                               epochs=epochs, steps_per_epoch=steps_per_epoch_train,
                               validation_data=valid_data, validation_steps=steps_per_epoch_valid,
                               callbacks=[checkpoint, statistics, visualize, tensorboard, csv_logger],
                               initial_epoch=start_epoch,
                               verbose=1,
                               use_multiprocessing=False,
                               workers=1)
    #K.set_learning_phase(0)
    #print(train_model.model.evaluate({'input':np.array(data[-2:]), 'labels':np.array(labels[-2:]), 'input_len':np.array(input_len[-2:]), 'label_len':np.array(output_len[-2:])}, {'ctc':np.zeros([2])}))


def train_avletter(train_path, valid_path, start_epoch=0, epochs=10, img_c=1, img_w=60, img_h=80, frames_n=25,
               absolute_max_string_len=1, batch_size=20, learning_rate=0.0001, output_dir='saved_models'):

    print ('Starting training...')
    train_num = len(glob.glob(train_path))
    valid_num = len(glob.glob(valid_path))

    train_data = LipNetDataGen(train_path, batch_size, img_c, img_w, img_h, frames_n, absolute_max_string_len)
    valid_data = LipNetDataGen(valid_path, batch_size, img_c, img_w, img_h, frames_n, absolute_max_string_len)

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    train_model = LipNetModel(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                              absolute_max_string_len=absolute_max_string_len, output_size=28)

    # Dummy function since the model calculates the loss
    train_model.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam,
                              metrics=[metrics.categorical_accuracy])

    # load weight if necessary
    if start_epoch > 0:
        weight_file = os.path.join(output_dir + '/', ('lipnet%02d.h5') % (start_epoch - 1))
        train_model.model.load_weights(weight_file)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    checkpoint = ModelCheckpoint(output_dir + '/' + "lipnet{epoch:02d}.h5", monitor='val_loss', verbose=0, mode='auto',
                                 period=50)

    # define callbacks
    statistics = Statistics(train_model, valid_data, decoder, 256, output_dir=os.path.join(output_dir + '/', 'stats'))
    visualize = Visualize(os.path.join(output_dir + '/' + 'visualise'), train_model, valid_data, decoder,
                          num_display_sentences=batch_size)
    tensorboard = TensorBoard(log_dir=os.path.join(output_dir + '/' + 'tensorboard'))
    csv_logger = CSVLogger(os.path.join(output_dir + '/' + 'tensorboard', "{}.csv".format('training')), separator=',',
                           append=True)

    # train_model.model.fit({'input':np.array(data[:-2]), 'labels':np.array(labels[:-2]), 'input_len':np.array(input_len[:-2]), 'label_len':np.array(output_len[:-2])}, {'ctc':np.zeros([len(data[:-2])])}, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_split=0.0)
    steps_per_epoch_train = train_num / batch_size
    steps_per_epoch_valid = valid_num / batch_size

    train_model.model.fit_generator(generator=train_data,
                                    epochs=epochs, steps_per_epoch=steps_per_epoch_train,
                                    validation_data=valid_data, validation_steps=steps_per_epoch_valid,
                                    callbacks=[checkpoint, statistics, visualize, tensorboard, csv_logger],
                                    initial_epoch=start_epoch,
                                    verbose=1,
                                    use_multiprocessing=False,
                                    workers=1)
    # K.set_learning_phase(0)
    # print(train_model.model.evaluate({'input':np.array(data[-2:]), 'labels':np.array(labels[-2:]), 'input_len':np.array(input_len[-2:]), 'label_len':np.array(output_len[-2:])}, {'ctc':np.zeros([2])}))

def main():

    if len(sys.argv) == 7:
        batch_size = int(sys.argv[2])
        start_epoch = int(sys.argv[3])
        epochs = int(sys.argv[4])
        learning_rate = float(sys.argv[5])
        output_dir = sys.argv[6]

    else:
        # Default
        print ('Resorting to default values')
        batch_size = 20
        start_epoch = 0
        epochs = 100
        learning_rate = 0.0001
        output_dir = 'saved_models'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if sys.argv[1] == 'grid':
        train_path = 'train_path/*.hdf5'
        valid_path = 'valid_path/*.hdf5'
        train_GRID(train_path=train_path, valid_path=valid_path, batch_size=batch_size,start_epoch=start_epoch, epochs=epochs, learning_rate=learning_rate,
                   output_dir=output_dir)

    elif sys.argv[1] == 'avletter':
        train_path = 'train_path_avletter/*.hdf5'
        valid_path = 'valid_path_avletter/*.hdf5'
        train_avletter(train_path=train_path, valid_path=valid_path,batch_size=batch_size, start_epoch=start_epoch, epochs=epochs, learning_rate=learning_rate,
                       output_dir=output_dir)


if __name__ == "__main__":
    main()
