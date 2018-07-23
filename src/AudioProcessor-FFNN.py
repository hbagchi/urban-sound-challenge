"""
Urban Sound Challenge - Sound Classification using Feed Forward Neural Network (FFNN)
- No Validation Set

@author: - Hitesh Bagchi
Version: 2.0
Date: 16th July 2018
"""
#------------------------------------------------------------------------------
# Python libraries import
#------------------------------------------------------------------------------
import os
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import SVG

from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from sklearn.preprocessing import LabelEncoder

# Audio library to extract audio features
import librosa, librosa.display

# Deep Learning library for model learning and prediction
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import model_to_dot
from keras.utils import np_utils

# %%
#------------------------------------------------------------------------------
# Data Exploration
#------------------------------------------------------------------------------
def load_sample_audios(train):
    sample = train.groupby('Class', as_index=False).agg(np.random.choice)
    raw_audios = []
    class_audios = []
    for i in range(0, len(sample)):
        x, sr = librosa.load('./data/train/' + str(sample.ID[i]) + '.wav')
        x = librosa.resample(x, sr, 22050)
        raw_audios.append(x)
        class_audios.append(sample.Class[i])
    return class_audios, raw_audios

# Plot Waveplot
def plot_waves(class_audios, raw_audios):
    for x, label in zip(raw_audios, class_audios):
        plt.figure(figsize=(8, 20))
        plt.subplot(10, 1, class_audios.index(label)+1)
        librosa.display.waveplot(x)
        plt.title(label)

# Plot Specgram
def plot_specgram(class_audios, raw_audios):
    for x, label in zip(raw_audios, class_audios):
        plt.figure(figsize=(8, 40))
        plt.subplot(10, 1, class_audios.index(label)+1)
        specgram(x, Fs=22050)
        plt.title(label)

# Plot log power specgram
def plot_log_power_specgram(class_audios, raw_audios):
    for x, label in zip(raw_audios, class_audios):
        plt.figure(figsize=(8, 40))
        plt.subplot(10, 1, class_audios.index(label)+1)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(x))**2, ref=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(label)

# Print raw mfcc for a randomly selected sample of each category
def extract_raw_mfcc(class_audios, raw_audios):
    for x, label in zip(raw_audios, class_audios):
        mfccs = librosa.feature.mfcc(y=x, n_mfcc=64).T
        print(label, mfccs, '\n')

# %%
# data directory and csv file should have the same name

DATA_PATH = 'train'

train = pd.read_csv('./data/' + DATA_PATH + '.csv')

class_audios, raw_audios = load_sample_audios(train)

# %%
extract_raw_mfcc(class_audios, raw_audios)

# %%
# Plot waveform, specgram and log power specgram

plot_waves(class_audios, raw_audios)
plot_specgram(class_audios, raw_audios)
plot_log_power_specgram(class_audios, raw_audios)

# %%
# check data distribution of training set
dist = train.Class.value_counts()
plt.figure(figsize=(8, 4))
plt.xticks(rotation=60)
plt.bar(dist.index, dist.values)

# %%
files_in_error = []

# Extracts audio features from data
def extract_features(row):
    # function to load files and extract features
    file_name = os.path.join(
            os.path.abspath('./data/'), 
            DATA_PATH, str(row.ID) + '.wav')

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
    features = train.apply(extract_features, axis=1)
    dump_features(features, features_train_file)
    train = train.assign(features=features.values)
else:
    features = pd.read_pickle('./features_train.pkl')
    train = train.assign(features=features.values)

# %%
y = np.array(train.loc[:, 'Class'])
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))

X = np.array(train.loc[:, 'features'])
X = np.vstack(X)

# Only MFCC features
# X = X[:, :64]

X -= X.mean(axis=0)
X /= X.std(axis=0)

# %%
# Build learning model
def buildModel(num_inputs, num_labels):
    
    # build model
    model = Sequential()

    # Input layer
    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.001),
                    input_shape=(num_inputs,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    # Hidden layer
    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    # Compile model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# %%
num_labels = y.shape[1] # Toral number of output labels
num_inputs = X.shape[1] # Total number 0f input variables

model_inst = buildModel(num_inputs, num_labels)    

SVG(model_to_dot(model_inst, 
                 show_shapes=True, 
                 show_layer_names=True).create(prog='dot', format='svg'))

# monitor training loss
early_stop = EarlyStopping(monitor='loss', min_delta=0, 
                       patience=2, verbose=0, mode='auto')

#######################################################################
# train set divided into 80% training set and 20% test set
# 80% train set = TRAIN SET, 20% train set = TEST SET
#######################################################################
x_train = X[:4348]
x_test = X[4348:]

y_train = y[:4348]
y_test = y[4348:]

test_df = train[4348:].copy()

# %%
history = model_inst.fit(x_train, y_train, batch_size=512, 
                         epochs=100, callbacks=[early_stop])

# %%
# Extract cost from history
history_dict = history.history
history_dict.keys()
loss_vals = history_dict['loss']
acc_vals = history_dict['acc']
epochs = history.epoch

# Plot training cost against epoch
train_loss_plot, = plt.plot(epochs, loss_vals, 'bo', label='Training Loss')
train_acc_plot, = plt.plot(epochs, acc_vals, 'r', label='Training Accuracy')

plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend([train_loss_plot, train_acc_plot], ["Training Loss", "Training Acc"])

# %%
# calculate predictions on hold out set (20% training set)
pred_test = model_inst.predict_classes(x_test)
pred_test_class = lb.inverse_transform(pred_test)

test_df.loc[:, "PredClass"] = pred_test_class

# drop the 'feature' column as it is not required
test_df = test_df.drop(columns=['features'], axis=1)

test_df['status'] = (test_df.Class == test_df.PredClass)

correct_match_count = test_df.groupby("status").size()[1]
incorrect_match_count = test_df.groupby("status").size()[0]

acc_ratio = correct_match_count/(correct_match_count + incorrect_match_count)
acc_ratio
