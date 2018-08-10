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
DATA_DIR = 'trainMini'

# Load TRAINING SET
TRAIN_CSV_PATH = './data/' + DATA_DIR + '.csv'  # Path where csv files are stored (train set)

train_df = pd.read_csv(TRAIN_CSV_PATH)

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

    file_name = os.path.join(os.path.abspath('./data/'), data_dir, str(row.ID) + '.wav')

    # handle exception to check for corrupted or invalid file
    try:
        X, sample_rate = librosa.load(file_name)

        for (start, end) in windows(X, window_size):
            if(len(X[start:end]) == window_size):
                signal = X[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)

                if 'Class' in row.index:
                    labels.append(row.Class) 

        log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames, 1)
        features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)

        for i in range(len(features)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    except Exception as e:
        print(file_name, e)
        return None

    return pd.Series([features, np.array(labels)])
#    return features

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
    extract_features_train = partial(extract_features, DATA_DIR)

    features_labels = train_df.apply(extract_features_train, axis=1)
    dump_features(features_labels, features_cnn_train_file)
else:
    features_labels = pd.read_pickle('./features_cnn_train.pkl')

features = features_labels[0]
labels = features_labels[1]
train_df = train_df.assign(features=features.values)

# %%
y = np.concatenate([item.tolist() for item in labels])
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))

X = np.array(train_df.loc[:, 'features'])
X = np.vstack(X)

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
DATA_DIR = 'testMini'

TEST_CSV_PATH = './data/' + DATA_DIR + '.csv'   # Path where csv files are stored (test set)

test_df = pd.read_csv(TEST_CSV_PATH)

features_cnn_test_file = Path("./features_cnn_test.pkl")

if not features_cnn_test_file.is_file():
    data_path = DATA_DIR
    extract_features_func = partial(extract_features, data_path)
    features_labels_test = test_df.apply(extract_features_func, axis=1)
    dump_features(features_labels_test, features_cnn_test_file)
else:
    features_labels_test = pd.read_pickle('./features_cnn_test.pkl')

features_test = features_labels_test[0]
test_df = test_df.assign(features=features_test.values)

# %%
X_test = np.array(test_df.loc[:, 'features'])
X_test = np.vstack(X_test)

# Only MFCC features
# X_test = X_test[:, :64]

# X_test -= train_mean    # training dataset mean is used for normalization
# X_test /= train_std     # training std mean is used for normalization

X_test.shape

# calculate predictions
predictions = model.predict_classes(X_test)
predict_class = lb.inverse_transform(predictions)

test_df['Class'] = predict_class
test_output = test_df.copy()

# %%

# drop the 'feature' column as it is not required for submission
test_output = test_output.drop(columns=['features'], axis=1)

test_output.to_csv('sub01-cnn.csv', index=False)
