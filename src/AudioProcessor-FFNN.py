"""
Urban Sound Challenge - Sound Classification using Feed Forward Neural Network (FFNN)
- Grid Search

@author: - Hitesh Bagchi
Version: 1.0
Date: 24th July 2018
"""
#------------------------------------------------------------------------------
# Python libraries import
#------------------------------------------------------------------------------
import os
from pathlib import Path

import numpy as np
import pandas as pd
from functools import partial

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# Audio library to extract audio features
import librosa, librosa.display

# Deep Learning library for model learning and prediction
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

# %%
#------------------------------------------------------------------------------
# Data Exploration Phase
#------------------------------------------------------------------------------
def load_sample_audios(train_df, data_path):
    sample = train_df.groupby('Class', as_index=False).agg(np.random.choice)
    raw_audios = []
    class_audios = []
    for i in range(0, len(sample)):
        x, sr = librosa.load(data_path + str(sample.ID[i]) + '.wav')
        x = librosa.resample(x, sr, 22050)
        raw_audios.append(x)
        class_audios.append(sample.Class[i])
    return class_audios, raw_audios

# ORIGINAL TEST SET
def load_plot_test_sample(test_df, test_data_path):
    file_index = str(int(test_df.sample(n=1).ID.values))    
    file_path = os.path.join(
            os.path.abspath(test_data_path), 
            file_index + '.wav')
    x, sr = librosa.load(file_path)
    x = librosa.resample(x, sr, 22050)

    plt.clf()
    plt.figure(figsize=(8, 20))
    plt.subplot(10, 1, 1)
    librosa.display.waveplot(x)
    plt.title(file_path)

# Plot Waveplot
def plot_waves(class_audios, raw_audios):
    plt.clf()
    for x, label in zip(raw_audios, class_audios):
        plt.figure(figsize=(8, 20))
        plt.subplot(10, 1, class_audios.index(label)+1)
        librosa.display.waveplot(x)
        plt.title(label)

# Plot Specgram
def plot_specgram(class_audios, raw_audios):
    plt.clf()
    for x, label in zip(raw_audios, class_audios):
        plt.figure(figsize=(8, 40))
        plt.subplot(10, 1, class_audios.index(label)+1)
        plt.specgram(x, Fs=22050)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title(label)

# Plot log power specgram
def plot_log_power_specgram(class_audios, raw_audios):
    plt.clf()
    for x, label in zip(raw_audios, class_audios):
        plt.figure(figsize=(8, 40))
        plt.subplot(10, 1, class_audios.index(label)+1)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(x))**2, ref=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(label)

# Print raw mfcc for a sample
def print_raw_mfcc(class_audios, raw_audios):
    for x, label in zip(raw_audios, class_audios):
        mfccs = librosa.feature.mfcc(y=x, n_mfcc=64).T
        print (label, mfccs, '\n')


# %%
# data directory and csv file should have the same name

TRAIN_CSV_PATH = './data/train.csv'   # Path where csv files are stored
TRAIN_DATA_PATH = './data/train/'     # Path where audio files are stored

train_df = pd.read_csv(TRAIN_CSV_PATH)

class_audios, raw_audios = load_sample_audios(train_df, TRAIN_DATA_PATH)

# %%
print_raw_mfcc(class_audios, raw_audios)

# %%
# Plot waveform, specgram and log power specgram

plot_waves(class_audios, raw_audios)
plot_specgram(class_audios, raw_audios)
plot_log_power_specgram(class_audios, raw_audios)

# %%
# Load and Plot TEST SET
TEST_CSV_PATH = './data/test.csv'   # Path where csv files are stored (test set)
TEST_DATA_PATH = './data/test/'     # Path where audio files are stored (test set)

test_df = pd.read_csv(TEST_CSV_PATH)
load_plot_test_sample(test_df, TEST_DATA_PATH)

# %%
# check data distribution of training set
dist = train_df.Class.value_counts()
plt.figure(figsize=(8, 4))
plt.xticks(rotation=60)
plt.bar(dist.index, dist.values)

# %%
files_in_error = []

# Extracts audio features from data
def extract_features(row, data_path):
    # function to load files and extract features
    file_name = os.path.join(os.path.abspath(data_path), str(row.ID) + '.wav')

    # handle exception to check if there isn't a file which is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

        stft = np.abs(librosa.stft(X))

        mfcc_count = 64

        mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=mfcc_count).T, axis=0)
        
        chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
        
        mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)
        
        tonnetz = np.mean(librosa.feature.tonnetz(
                y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        
        contrast = np.mean(librosa.feature.spectral_contrast(
                S=stft, sr=sample_rate).T, axis=0)

    except Exception as e:
        print(file_name, e)
        files_in_error.append(file_name)
        return None

    features = np.hstack([mfccs, chroma, mel, tonnetz, contrast])

    return features

# %%
def dump_features(features, features_filename):
    features_df = pd.DataFrame(features)
    features_df.to_pickle(features_filename)

# %%
features_train_file = Path("./features_train.pkl")

if not features_train_file.is_file():
    data_path = TRAIN_DATA_PATH
    extract_features_func = partial(extract_features, data_path)
    features = train_df.apply(extract_features_func, axis=1)
    dump_features(features, features_train_file)
    train_df = train_df.assign(features=features.values)
else:
    features = pd.read_pickle('./features_train.pkl')
    train_df = train_df.assign(features=features.values)

# %%
y = np.array(train_df.loc[:, 'Class'])
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))

X = np.array(train_df.loc[:, 'features'])
X = np.vstack(X)

# Only MFCC features
# X = X[:, :64]

train_mean = X.mean(axis=0)
train_std = X.std(axis=0)

X -= train_mean
X /= train_std

# %%
# Build learning model
def build_model(num_input_vars,
                num_target_labels,
                dropout_rate=0.2,
                optimizer='rmsprop',
                neurons=128,
                activation='relu',
                regularizer_l2=0.01):
    
    # build model
    model = Sequential()

    # Input layer
    model.add(Dense(neurons, input_shape=(num_inputs,),
                    kernel_regularizer=regularizers.l2(regularizer_l2)))
    
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    
    # Hidden layer
    model.add(Dense(neurons, kernel_regularizer=regularizers.l2(regularizer_l2)))
    
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    # Compile model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# %%
num_labels = y.shape[1] # Toral number of output labels
num_inputs = X.shape[1] # Total number 0f input variables

# create model
model_inst = KerasClassifier(build_fn=build_model, 
                             num_input_vars=num_inputs, 
                             num_target_labels=num_inputs)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, 
                       patience=2, verbose=0, mode='auto')

# 80% training set, 20% validation set

x_val = X[4348:]
X = X[:4348]

y_val = y[4348:]
y = y[:4348]

# %%
#################################################
# Grid Search Section
#################################################

# define the hyperparameters for grid search
batch_size = [128, 256, 512]
dropout_rate = [0.2, 0.3, 0.5]
optimizer = ['rmsprop', 'Adam']
neurons = [96, 128, 156]
activation = ['relu']
regularizer_l2 = [0.01, 0.1, 0.3]
epochs = 200

param_grid = dict(batch_size=batch_size, 
                  dropout_rate=dropout_rate,
                  optimizer=optimizer,
                  neurons=neurons,
                  activation=activation,
                  regularizer_l2=regularizer_l2)

grid_model_inst = GridSearchCV(estimator=model_inst, param_grid=param_grid)

grid_result = grid_model_inst.fit(X, y, epochs=epochs,
                                  validation_data=(x_val, y_val), 
                                  callbacks=[early_stop], 
                                  verbose=1)

# summarize results
grid_result.best_score_, grid_result.best_params_

# %%
''' Get Hyperparameters from Grid Search or set the hyperparameters manually
    if Grid Search is not run to save time
'''

if 'grid_result' in globals():
    activation = grid_result.best_params_.get('activation')
    batch_size = grid_result.best_params_.get('batch_size')
    dropout_rate = grid_result.best_params_.get('dropout_rate')
    neurons = grid_result.best_params_.get('neurons')
    optimizer = grid_result.best_params_.get('optimizer')
    regularizer_l2 = grid_result.best_params_.get('regularizer_l2')
else:
    activation = 'relu'
    batch_size = 512
    dropout_rate = 0.3
    neurons = 128
    optimizer = 'rmsprop'
    regularizer_l2 = 0.01
    epochs = 200

# create model
grid_model = KerasClassifier(build_fn=build_model, 
                             num_input_vars=num_inputs, 
                             num_target_labels=num_inputs)

grid_model = build_model(num_inputs,
                         num_labels,
                         dropout_rate=dropout_rate,
                         optimizer=optimizer,
                         neurons=neurons,
                         activation=activation,
                         regularizer_l2=regularizer_l2)
grid_model.summary()

history = grid_model.fit(X, y, batch_size=batch_size, epochs=epochs,
                         validation_data=(x_val, y_val),
                         callbacks=[early_stop])

# %%
# Extract cost from history

history_dict = history.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = history.epoch

# Plot training and validation cost against epoch

train_loss_plot, = plt.plot(epochs, loss_values, 'bo', label='Training Loss')
val_loss_plot, = plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend([train_loss_plot, val_loss_plot], ["Training Loss", "Validation Loss"])

# %%

# data directory and csv file should have the same name

test_df = pd.read_csv(TEST_CSV_PATH)

features_test_file = Path("./features_test.pkl")

if not features_test_file.is_file():
    data_path = TEST_DATA_PATH
    extract_features_func = partial(extract_features, data_path)
    features = test_df.apply(extract_features_func, axis=1)
    dump_features(features, features_test_file)
    test_df = test_df.assign(features=features.values)
else:
    features = pd.read_pickle('./features_test.pkl')
    test_df = test_df.assign(features=features.values)

# %%
X_test = np.array(test_df.loc[:, 'features'])
X_test = np.vstack(X_test)

# Only MFCC features
# X_test = X_test[:, :64]

X_test -= train_mean    # training dataset mean is used for normalization
X_test /= train_std     # training std mean is used for normalization

X_test.shape

# calculate predictions
predictions = grid_model.predict_classes(X_test)
predict_class = lb.inverse_transform(predictions)

test_df['Class'] = predict_class
test_output = test_df.copy()

# drop the 'feature' column as it is not required for submission
test_output = test_output.drop(columns=['features'], axis=1)

test_output.to_csv('sub01.csv', index=False)
