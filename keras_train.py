import os, cv2, random
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import h5py
import matplotlib.pyplot as plt
from matplotlib import ticker
#import seaborn as sns
#%matplotlib inline 

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K
TRAIN_DIR = 'C:/Users/main/desktop/train/'
TEST_DIR = 'C:/Users/main/desktop/test_stg1/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ROWS = 720
COLS = 1280
CHANNELS = 3

optimizer = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'
X_train = np.load('C:/Users/main/desktop/X_train.npy')
#X_valid = np.load('C:/Users/main/desktop/X_valid')
y_train =np.load('C:/Users/main/desktop/y_train.npy')
#np.save('C:/Users/main/desktop/y_valid',y_valid)

model = Sequential()

model.add(Activation(activation='relu', input_shape=(ROWS, COLS, CHANNELS)))


model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(4, 4), dim_ordering='tf'))

model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(4, 4), dim_ordering='tf'))

model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(4, 4), dim_ordering='tf'))

model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(4, 4), dim_ordering='tf'))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(FISH_CLASSES)))
model.add(Activation('sigmoid'))

model.compile(loss=objective, optimizer=optimizer)


early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')        
        
model.fit(X_train, y_train, batch_size=8, nb_epoch=1,
              validation_split=0.2, verbose=1, shuffle=True, callbacks=[early_stopping])

##saving model to reload
model.save('C:/Users/main/desktop/fishtrained.h5')
