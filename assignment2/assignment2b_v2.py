# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="empirical-grain"
# # Assignment2b - V.2

# %% id="invalid-pathology"
import os
import sys, time, random, gc, socket
import logging
from collections import Counter
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.exposure import equalize_adapthist

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import LSTM, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file

# %% colab={"base_uri": "https://localhost:8080/"} id="demographic-costa" outputId="a4d38e41-fb50-44af-c0c4-89717f70fa95"
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

# %% id="induced-weight"
logger = logging.getLogger('my_happy_logger')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

# %% id="valid-bailey"
# GPU memory fix + Mac workaround
if not IS_COLAB:
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# %% id="assisted-princeton"
os.makedirs('textgen-figures', exist_ok=True)

# %% id="significant-sequence"
np.random.seed(42)
tf.random.set_seed(42)


# %% id="manual-record"
class assign2:
    
    def __init__(self, filename='nietzsche.txt', 
                 filepath='https://s3.amazonaws.com/text-datasets/nietzsche.txt'):
        
        # saving static variables in __init__
        self.text = self.download_file(filename, filepath)
        self.chars = dict(sorted(Counter(self.text).items(), key=lambda x:x[1], reverse=True)).keys()
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        
        # dyanmic variables - these are the variables that need to be changed via clear() after each modeling process
        self.charmap = None
        self.hmap = None
        self.model = None
        
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

        return X, y

    def build_one_layer_lstm(self, maxlen=40, hidden_units=128):
        input_shape = (maxlen, len(self.chars))
        model = Sequential()
        model.add(LSTM(hidden_units, input_shape=input_shape))
        model.add(Dense(len(self.chars)))
        model.add(Activation('softmax'))
        optimizer = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def build_cnn_v1(self, maxlen=40):
        input_shape_ = (maxlen, len(self.chars)) + (1,)
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(2, len(self.chars)), activation='relu', input_shape=input_shape_))
        model.add(BatchNormalization())
        model.add(Conv2D(256, kernel_size=(5, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, kernel_size=(5, 1), activation='relu'))
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
                viz[i,:] = s

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

            plt.imshow(equalize_adapthist(np.log(viz.T+1)), cmap='gray')
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
        
    def main(self, num_iter: int=600, freq: int=5, modeltype: str='lstm',
             save_weights: bool=True, load_weights: bool=False, batch_size: int=128,
             epochs: int=1, maxlen: int=40, step: int=1, temperature: list=[0.2, 0.5, 1.0, 1.2]):
        
        modelname = 'textgen-figures/' + 'model-text-' + modeltype + '.h5'
        
        # step 1: preprocess
        X, y = self.vectorize(maxlen, step)
        if modeltype == 'cnn':
            X = np.expand_dims(X, axis=3)
        
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

            print("Total training time: %0.2f min, epoch %d" % ((time.time() - tm)/60.0, iteration))
            print("Average time per epoch: %0.2f s" % ((time.time() - tm)/iteration))

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
                
    def clear(self):
        self.charmap = None
        self.hmap = None
        self.model = None


# %% [markdown] id="spanish-temple"
# ## Problem 1
#
# General steps:
# 1) Instantiate class once (unless you want to change the dataset) \
# 2) configure the model and input shape (either use example model with different parameters, or define a custom model via obj.model = Sequential(), etc.) \
# 3) call the main method with the right arguments \
# 4) clear variables
#
# For example, here's what you do for Problem 1:

# %% id="unsigned-yield"
obj = assign2()
obj.model = obj.build_one_layer_lstm(maxlen=40)
obj.main(num_iter=3, freq=2, modeltype='lstm')
obj.clear()

# %% [markdown] id="answering-parade"
# ## Problem 2
#
# (say you want to change the default temperature values)

# %% id="homeless-event"
obj.model = obj.build_one_layer_lstm(maxlen=40)
obj.main(temperature=[0.1, 0.2, 0.3, 0.4])
obj.clear()

# %% [markdown] id="nearby-dealer"
# ## Problem 4
#
# Here's how you can configure your model outside the class (alternatively, you can define a new method inside the class):

# %% id="generous-assault"
input_shape = (40, len(obj.chars))
obj.model = Sequential()
obj.model.add(LSTM(64, input_shape=input_shape))
obj.model.add(Dense(len(obj.chars)))
obj.model.add(Activation('softmax'))
optimizer = Adam(lr=0.001)
obj.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

obj.main()
obj.clear()

# this is equivalent to the following:
obj.model = obj.build_one_layer_lstm(maxlen=40, hidden_units=64)
obj.main()
obj.clear()

# %% [markdown] id="8sMAunVLVoDe"
# This is just to demonstrate another way to write model with `keras.model.Sequential`. Personally, I think this method allows us to write the model code more concise.

# %% colab={"base_uri": "https://localhost:8080/"} id="TckdAyd8UwR3" outputId="31d59265-6ac9-48c1-e402-d0226d59c6f4"
input_shape = (40, len(obj.chars))
obj.model = Sequential([
    LSTM(64, input_shape=input_shape),
    Dense(len(obj.chars), activation='softmax')
    ])
optimizer = Adam(lr=0.001)
obj.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

obj.main(num_iter=1, modeltype='lstm')
obj.clear()

# %% [markdown] id="d4AP3DmyQ1xJ"
# ## Problem 5 - CNN

# %% id="y5QwH5ITQ1EK"
obj = assign2()
obj.model = obj.build_cnn_v1()
obj.main(num_iter=3, freq=2, modeltype='cnn')

# %% id="TNRrdtWkRNSZ"
model_name = 'test_cnn'
obj.model.save_weights(f'{model_name}.h5')
obj.clear()

# %% id="y1Gebd9OTtV2"
