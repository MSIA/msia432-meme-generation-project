import os
import sys, time, random, gc, socket
import logging
from collections import Counter
from typing import Tuple
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.exposure import equalize_adapthist

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import LSTM, Conv2D, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file

try:
    # tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
    IS_COLAB = True
except Exception:
    IS_COLAB = False

assert tf.__version__ >= "2.0"

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
else:
  print('NUM GPUS:', len(tf.config.list_physical_devices('GPU')))

logger = logging.getLogger('my_happy_logger')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

if not IS_COLAB:
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# %% id="assisted-princeton"
os.makedirs('textgen-figures', exist_ok=True)

# %% id="significant-sequence"
np.random.seed(42)
tf.random.set_seed(42)


class assign2:

    def __init__(self, filename='nietzsche.txt',
                 filepath='https://s3.amazonaws.com/text-datasets/nietzsche.txt'):

        # saving static variables in __init__
        self.text = self.download_file(filename, filepath)
        self.chars = dict(sorted(Counter(self.text).items(), key=lambda x: x[1], reverse=True)).keys()
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        # dyanmic variables - these are the variables that need to be changed via clear() after each modeling process
        self.charmap = None
        self.hmap = None
        self.model = None
        self.final_loss = None
        self.total_time = None

    def download_file(self, filename, filepath):
        path = get_file(filename, filepath)
        text = open(path).read().lower()
        logger.debug(f'corpus length: {len(text)}')
        return text

    def tokenize(self, maxlen, step):

        # break text into equal-length chunks
        char_chunks = [self.text[i: i + maxlen] for i in range(0, len(self.text) - maxlen, step)]
        next_char = [self.text[i + maxlen] for i in range(0, len(self.text) - maxlen, step)]
        logger.debug(f'nb sequences: {len(char_chunks)}')

        return char_chunks, next_char

    def vectorize(self, maxlen, step):
        char_chunks, next_char = self.tokenize(maxlen, step)
        X = np.zeros((len(char_chunks), maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(char_chunks), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(char_chunks):
            for t, char in enumerate(sentence):
                X[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_char[i]]] = 1

        ## return X, y
        return np.expand_dims(X, axis=3), y

    def build_one_layer_lstm(self, maxlen=40, hidden_units=128):
        input_shape = (maxlen, len(self.chars))
        model = Sequential()
        model.add(LSTM(hidden_units, input_shape=input_shape))
        model.add(Dense(len(self.chars)))
        model.add(Activation('softmax'))
        optimizer = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def build_one_layer_gru(self, maxlen=40, hidden_units=128):
        input_shape = (maxlen, len(self.chars))
        model = Sequential()
        model.add(GRU(hidden_units, input_shape=input_shape))
        model.add(Dense(len(self.chars)))
        model.add(Activation('softmax'))
        optimizer = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def build_cnn_v1(self, maxlen=40, filter_size=5, filter_number=128):
        input_shape_ = (maxlen, len(self.chars)) + (1,)
        model = Sequential()
        model.add(Conv2D(filter_number, kernel_size=(2, len(self.chars)), activation='relu', input_shape=input_shape_))
        model.add(BatchNormalization())
        model.add(Conv2D(filter_number * 2, kernel_size=(filter_size, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filter_number * 4, kernel_size=(filter_size, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(len(self.chars)))
        model.add(Activation('softmax'))
        optimizer = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def build_cnn_v2(self, maxlen=40, filter_size=5, filter_number=128):
        input_shape_ = (maxlen, len(self.chars)) + (1,)
        model = Sequential()
        model.add(Conv2D(filter_number, kernel_size=(2, len(self.chars)), activation='relu', input_shape=input_shape_))
        model.add(BatchNormalization())
        model.add(Conv2D(filter_number * 2, kernel_size=(filter_size, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filter_number * 4, kernel_size=(filter_size, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filter_number * 8, kernel_size=(filter_size, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(len(self.chars)))
        model.add(Activation('softmax'))
        optimizer = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    # Load existing checkpoint
    def load_checkpoint(self, modelname):
        if os.path.exists(modelname):
            logger.info('Try loading model: %s', modelname)
        try:
            self.model.load_weights(modelname)
            logger.info('Loaded model: %s', modelname)
        except Exception:
            logger.error('Error in model, not loading...')

    def generate_text(self, maxlen, input_shape, temperature):
        start_index = random.randint(0, len(self.text) - maxlen - 1)
        for diversity in temperature:
            print()
            print('----- diversity:', diversity)
            viz = np.ndarray((400, len(self.chars)))

            generated = ''
            sentence = self.text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x = np.zeros((1,) + input_shape)

                for t, char in enumerate(sentence):
                    x[0, t, self.char_indices[char]] = 1.

                preds = self.model.predict(x, verbose=0)[0]
                s = self.sample(preds, diversity)
                viz[i, :] = s

                if i < 5:
                    plt.bar(list(range(len(s))), -np.log(s))
                    plt.show()
                s = np.random.multinomial(1, s, 1)
                next_index = np.argmax(s)
                next_char = self.indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()

            print()
            self.viz_heatmap_chars(viz[:30])
            plt.show(), plt.close()

            plt.imshow(equalize_adapthist(np.log(viz.T + 1)), cmap='gray')
            plt.title('Visualization of one-hot characters')
            plt.grid(False)
            plt.show()
            plt.close()

            # used by generate_text

    def sample(self, preds, temperature):
        """helper function to sample an index from a probability array"""
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return preds

    # used by generate_text
    def viz_heatmap_chars(self, data, topk=5):
        """Helpers for visualization"""
        self.charmap = np.zeros((len(data), topk), dtype='str')
        self.hmap = np.zeros((len(data), topk))
        for d in range(len(data)):
            r = np.argsort(-data[d])[:topk]
            h = data[d][r]
            self.hmap[d] = h
            r = list(map(lambda x: self.indices_char[x], r))
            self.charmap[d] = np.array(r)
        sns.heatmap(self.hmap.T, annot=self.charmap.T, fmt='', cmap='Wistia')
        plt.title('Character heatmap')

    def main(self, num_iter: int = 600, freq: int = 5, modeltype: str = 'lstm',
             save_weights: bool = True, load_weights: bool = False, batch_size: int = 128,
             epochs: int = 1, maxlen: int = 40, step: int = 1, temperature: list = [0.2, 0.5, 1.0, 1.2],
             modelname: str = None, filter: int = 5):

        if modelname is None:
            modelname = 'textgen-figures/' + 'model-text-' + modeltype + '.h5'

        # step 1: preprocess
        X, y = self.vectorize(maxlen, step)

        # optional step: load weights
        if load_weights:
            self.load_checkpoint(modelname)

        # step 2: train the model, output generated text after each iteration
        tm = time.time()
        losses = []
        for iteration in range(1, num_iter + 1):
            print()
            print('-' * 50)
            print('Iteration', iteration)

            history = self.model.fit(X, y, batch_size, epochs)
            losses.append(history.history['loss'])

            print("Average time per epoch: %0.2f s" % ((time.time() - tm) / iteration))

            if iteration % freq == 0:
                if save_weights:
                    self.model.save_weights(modelname)
                input_shape = self.model.layers[0].input_shape[1:]
                self.generate_text(maxlen, input_shape, temperature)
                plt.plot(losses)
                plt.yscale('log')
                plt.title('Loss')
                plt.show()
                plt.close()

        final_loss = losses[-1]
        self.final_loss = final_loss
        total_time = (time.time() - tm) / num_iter
        self.total_time = total_time

    def clear(self):
        self.charmap = None
        self.hmap = None
        self.model = None

filter_size_collection = []
filter_number_collection = []
loss_collection = []
time_collection = []

for filter_size in [7,9,11]:
    for filter_number in [64, 128, 256]:
        obj = assign2()
        obj.model = obj.build_cnn_v1(maxlen = 40, filter_size = filter_size, filter_number = filter_number)
        obj.main(num_iter = 30, freq = 30, modeltype = 'cnn')
        filter_size_collection.append(filter_size)
        filter_number_collection.append(filter_number)
        loss_collection.append(obj.final_loss)
        time_collection.append(obj.total_time)
        obj.clear()

# Then we want to test if adding another layer makes significant difference
filter_size_collection = []
filter_number_collection = []
loss_collection = []
time_collection = []

for filter_size in [7, 9]:
    for filter_number in [32, 64, 128]:
        obj = assign2()
        obj.model = obj.build_cnn_v2(maxlen = 40, filter_size = filter_size, filter_number = filter_number)
        obj.main(num_iter = 30, freq = 40, modeltype = 'cnn')
        filter_size_collection.append(filter_size)
        filter_number_collection.append(filter_number)
        loss_collection.append(obj.final_loss)
        time_collection.append(obj.total_time)
        obj.clear()

# from all the results above, cnn1 with original layer, filter size of 7 and filter number of 256 provides the best result of losses after 30 iterations