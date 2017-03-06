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
from keras.models import load_model
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


def get_images(fish):
    """Load files from train folder"""
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir)]
    return images
def read_image(src):
    """Read and resize individual images"""
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    print(src)
    im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
    return im

cheese = load_model('C:/Users/main/desktop/fishtrained.h5')
model = cheese
X_train = np.load('C:/Users/main/desktop/X_train.npy')
X_valid = np.load('C:/Users/main/desktop/X_valid.npy')
y_train = np.load('C:/Users/main/desktop/y_train.npy')
y_valid = np.load('C:/Users/main/desktop/y_valid.npy')
preds = model.predict(X_valid, batch_size=8, verbose=1)
print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))

test_files = [im for im in os.listdir(TEST_DIR)] 

test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(test_files):
	test[i] = read_image(TEST_DIR+im)

test_preds = model.predict(test, batch_size=8, verbose=1)


submission = pd.DataFrame(test_preds, columns=FISH_CLASSES)
submission.insert(0, 'image', test_files)
submission.head()
