# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: msia432
#     language: python
#     name: msia432
# ---

# %% [markdown]
# # Assignment2b - V.2

# %%
import os
import sys, time, random, gc, socket
import logging
from collections import Counter
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from skimage.exposure import equalize_adapthist

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import LSTM, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file

# %%
try:
    # %tensorflow_version only exists in Colab.
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

# %%
logger = logging.getLogger('my_happy_logger')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

# %%
# GPU memory fix + Mac workaround
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# %%
os.makedirs('textgen-figures', exist_ok=True)

# %%
np.random.seed(42)
tf.random.set_seed(42)


# %%
class assign2:
    
    def __init__(self, filename='nietzsche.txt', 
                 filepath='https://s3.amazonaws.com/text-datasets/nietzsche.txt'):
        
        # saving static variables in __init__
        self.text = self.get_file(filename, filepath)
        self.chars = dict(sorted(Counter(self.text).items(), key=lambda x:x[1], reverse=True)).keys()
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        
        # dyanmic variables - these are the variables that need to be changed via clear() after each modeling process
        self.charmap = None
        self.hmap = None
        self.input_shape = None
        self.model = None
        
    def get_file(self, filename, filepath):
        path = get_file(filename, filepath)
        text = open(path).read().lower()
        logger.debug('corpus length: ' + str(len(text)))
        return text
        
    def tokenize(self, maxlen, step):
        
        # break text into equal-length chunks
        char_chunks = [self.text[i: i + maxlen] for i in range(0, len(self.text) - maxlen, step)]
        next_char = [self.text[i + maxlen] for i in range(0, len(self.text) - maxlen, step)]
        logger.debug('nb sequences:' + len(char_chunks)))
        
        return char_chunks, next_char
        
    def vectorize(self, maxlen, step, modeltype):
        char_chunks, next_char = self.tokenize(maxlen, step)
        X = np.zeros((len(char_chunks), maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(char_chunks), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(char_chunks):
            for t, char in enumerate(char_chunks):
                X[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_char[i]]] = 1

        if modeltype == 'cnn':
            return np.expand_dims(X, axis=3), y
        else:
            return X, y
     
    def example_model(self, maxlen, modeltype):
       
        #%% build the model: a single LSTM or CNN
        print('Build model...')
        self.input_shape = (maxlen, len(self.chars)) ## NEED TO CHANGE MAXLEN

        model = Sequential()
        if modeltype == 'lstm':
            # Note that this is a single-layer LSTM with 128 hidden units
            model.add(LSTM(128, input_shape=self.input_shape))
        elif modeltype == 'cnn':
            # Note that this is a simple CNN with 128, 256 and 512 hidden units. The receptive field is small.
            self.input_shape += (1,)
            model.add(Conv2D(128, kernel_size=(2, len(chars)), activation='relu', input_shape=self.input_shape))
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
    
    def generate_text(self, maxlen, temperature):
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
                x = np.zeros((1,) + self.input_shape)

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
            plt.show(), plt.close()        
        
    # used by generate_text
    def sample(preds, temperature):
        """helper function to sample an index from a probability array"""
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return preds

    # used by generate_text
    def viz_heatmap_chars(data, topk=5):
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
        
    def main(self, save_weights=True, load_weights=False, iteration=600, batch_size=128, epochs=1,
            maxlen=40, step=1, checkpoint_freq=5, modeltype='lstm', temperature=[0.2, 0.5, 1.0, 1.2]):
        
        modelname = 'model-text-' + modeltype + '.h5'
        
        # step 1: preprocess
        X, y = self.vectorize(maxlen, step, modeltype)
        
        # optional step: load weights
        if load_weights == True:
            if os.path.exists(modelname): 
                print('Loading model:' + modelname)
                try: self.model.load_weights(modelname)
                except: logger.error('Error in model, not loading...')
                
        # step 2: train the model, output generated text after each iteration
        tm = time.time()
        losses = []
        for i in range(1, iteration + 1):
            print()
            print('-' * 50)
            print('Iteration', i)

            loss = self.model.fit(X, y, batch_size, epochs)
            losses.append(loss.history['loss'])

            print("Total training time: %0.2f min, epoch %d" % ((time.time() - tm)/60.0, i))
            print("Average time per epoch: %0.2f s" % ((time.time() - tm)/i))

            if i % checkpoint_freq == 0: 
                if save_weights == True:
                    self.model.save_weights(modelname)
                self.generate_text(maxlen, temperature)
                plt.plot(losses)
                plt.yscale('log')
                plt.title('Loss')
                plt.show(), plt.close()
                
    def clear(self):
        self.charmap = None
        self.hmap = None
        self.input_shape = None
        self.model = None


# %% [markdown]
# ## Problem 1
#
# General steps:
# 1) Instantiate class once (unless you want to change the dataset) \
# 2) configure the model and input shape (either use example model with different parameters, or define a custom model via obj.model = Sequential(), etc.) \
# 3) call the main method with the right arguments \
# 4) clear variables
#
# For example, here's what you do for Problem 1:

# %%
obj = assign2()
obj.model = obj.example_model(maxlen=40, modeltype='lstm')
obj.main()
obj.clear()

# %% [markdown]
# ## Problem 2
#
# (say you want to change the default temperature values)

# %%
obj.model = obj.example_model(maxlen=40, modeltype='lstm')
obj.main(temperature=[0.1, 0.2, 0.3, 0.4])
obj.clear()

# %% [markdown]
# ## Problem 4
#
# Here's how you can configure your model outside the class:

# %%
obj.input_shape = (40, len(obj.chars))
obj.model = Sequential()
obj.model.add(LSTM(64, input_shape=obj.input_shape))
obj.model.add(Dense(len(self.chars)))
obj.model.add(Activation('softmax'))
optimizer = Adam(lr=0.001)
obj.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

obj.main()
obj.clear()
