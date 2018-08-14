"""
Urban Sound Challenge - Sound Classification using Convolutional Neural Network (CNN)
@author: - Hitesh Bagchi
Version: 1.0
Date: 06th August 2018
"""
#------------------------------------------------------------------------------
# Python libraries import
#------------------------------------------------------------------------------
import os
from pathlib import Path

import librosa
import pandas as pd
import numpy as np
from functools import partial
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

from keras.utils import np_utils

# %%
TR_DATA_DIR = 'train'
TST_DATA_DIR = 'test'

# Load TRAINING SET
# Path where csv files are stored (train set)
TRAIN_CSV_PATH = './data/' + TR_DATA_DIR + '.csv'

train_df = pd.read_csv(TRAIN_CSV_PATH)

# %%
#------------------------------------------------------------------------------
# function to repeat the audio if it has less than 20480 (512*40) samples 
#------------------------------------------------------------------------------
def repeat_sample(data):
    data_add = data
    while (len(data_add) < 20480):
        data_add = np.append(data_add, data)
    return data_add

# %%
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size/2)

# %%
#------------------------------------------------------------------------------
# function to load files and extract features
#------------------------------------------------------------------------------
def extract_features(data_dir, row, bands = 60, frames = 41, file_ext="*.wav"):

    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    ID_segs = []

    file_name = os.path.join(os.path.abspath('./data/'), data_dir, str(row.ID) + '.wav')

    # handle exception to check for corrupted or invalid file
    try:
        X, sample_rate = librosa.load(file_name)
        
        X = repeat_sample(X)
        
        for (start, end) in windows(X, window_size):
            if(len(X[start:end]) == window_size):
                signal = X[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)

                if 'Class' in row.index:
                    labels.append(row.Class) 
                
                ID_segs.append(row.ID)

        log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames, 1)
        features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)

        for i in range(len(features)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    except Exception as e:
        print(file_name, e)
        return None

    return pd.Series([np.array(ID_segs), features, np.array(labels)])

# %%
def dump_features(features, features_filename):
    features_df = pd.DataFrame(features)
    features_df.to_pickle(features_filename)

# %%
#------------------------------------------------------------------------------
# Load training data and extract features
#------------------------------------------------------------------------------
features_cnn_train_file = Path("./features_cnn_train.pkl")

if not features_cnn_train_file.is_file():
    extract_features_train = partial(extract_features, TR_DATA_DIR)

    features_labels = train_df.apply(extract_features_train, axis=1)
    dump_features(features_labels, features_cnn_train_file)
else:
    features_labels = pd.read_pickle('./features_cnn_train.pkl')

ID = features_labels[0]
features = features_labels[1]
labels = features_labels[2]
train_df = train_df.assign(features=features.values)

# %%
y = np.concatenate([item.tolist() for item in labels])
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))

X = np.array(train_df.loc[:, 'features'])
X = np.vstack(X)

train_mean = X.mean(axis=0)
train_std = X.std(axis=0)

X -= train_mean
X /= train_std

# %%
num_labels = y.shape[1] # Toral number of output labels
num_inputs = X.shape[1] # Total number 0f input variables

frames = 41
bands = 60
num_channels = 2

# build linear model
model = Sequential()

# will use filters of size 2x2 
f_size = 2

# first layer applies 32 convolution filters 
# input: 60x41 data frames with 2 channels => (60,41,2) tensors
model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(bands, frames, num_channels)))
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

# next layer applies 64 convolution filters
model.add(Conv2D(64, kernel_size=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# flatten output into a single dimension 
# Keras will do the shape inference automatically
model.add(Flatten())

# then a fully connected NN layer
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# finally, an output layer with one node per class
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# use the Adam optimiser
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# now compile the model, Keras will take care of the Tensorflow boilerplate
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

# %%
# for quicker training, just using one epoch, you can experiment with more
epochs = 10
model.fit(X, y, validation_split=0.2, batch_size=32, epochs=epochs)

# %%
# data directory and csv file should have the same name

# Load TEST SET
# Path where csv files are stored (test set)
TEST_CSV_PATH = './data/' + TST_DATA_DIR + '.csv'

test_df = pd.read_csv(TEST_CSV_PATH)

features_cnn_test_file = Path("./features_cnn_test.pkl")

if not features_cnn_test_file.is_file():
    data_path = TST_DATA_DIR
    extract_features_func = partial(extract_features, data_path)
    features_labels_test = test_df.apply(extract_features_func, axis=1)
    dump_features(features_labels_test, features_cnn_test_file)
else:
    features_labels_test = pd.read_pickle('./features_cnn_test.pkl')
    
ID_segs_tst = features_labels_test[0]
ID_segs_tst = np.concatenate([item.tolist() for item in ID_segs_tst])

features_test = features_labels_test[1]
test_df = test_df.assign(features=features_test.values)

# %%
X_test = np.array(test_df.loc[:, 'features'])
X_test = np.vstack(X_test)

# Only MFCC features
# X_test = X_test[:, :64]

X_test -= train_mean    # training dataset mean is used for normalization
X_test /= train_std     # training std mean is used for normalization

X_test.shape

# calculate predictions
predictions = model.predict_classes(X_test)
predict_class = lb.inverse_transform(predictions)

# Aggregate segment predictions by ID and select the Class with max count

id_class_df = pd.DataFrame(data=[ID_segs_tst, predict_class]).T
id_class_df = id_class_df.rename(columns={0:"ID", 1:"Class"})

seg_merged_df = id_class_df.groupby(['ID'], as_index=False)['Class'].max()

test_df['Class'] = seg_merged_df['Class']
test_output = test_df.copy()

# %%

# drop the 'feature' column as it is not required for submission
test_output = test_output.drop(columns=['features'], axis=1)

test_output.to_csv('sub01-cnn.csv', index=False)

# %%