"""
Urban Sound Challenge - Sound Classification using Feed Forward Neural Network (FFNN)
@author: - Hitesh Bagchi
Version: 1.0
Date: 25th June 2018
"""
#-------------------------------------------------------------------------------
# Python libraries import
#-------------------------------------------------------------------------------
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from matplotlib.pyplot import specgram
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score

from IPython.display import SVG

import librosa, librosa.display

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import Callback
from keras.utils.vis_utils import model_to_dot
from keras.utils import np_utils

# %%
#-------------------------------------------------------------------------------
# Data Exploration
#-------------------------------------------------------------------------------
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

# %%

# data directory and csv file should have the same name
DATA_PATH = 'train'

train = pd.read_csv('./data/' + DATA_PATH + '.csv')

class_audios, raw_audios = load_sample_audios(train)
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
    file_name = os.path.join(os.path.abspath('./data/'), DATA_PATH, str(row.ID) + '.wav')

    # handle exception to check if there isn't a file which is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

        stft = np.abs(librosa.stft(X))

        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=64).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                                  sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

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
class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_epochs = []

    def on_epoch_end(self, epoch, logs={}):

        self.val_epochs.append(epoch)

        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        _val_f1 = f1_score(val_targ, val_predict, average="micro")
        _val_recall = recall_score(val_targ, val_predict, average="micro")
        _val_precision = precision_score(val_targ, val_predict, average="micro")

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        print(" — val_f1: %f — val_precision: %f — val_recall: %f"
              %(_val_f1, _val_precision, _val_recall), '\n')

metrics = Metrics()

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

# %%
num_labels = y.shape[1] # Toral number of output labels
num_inputs = X.shape[1] # Total number 0f input variables

# build model
model = Sequential()

# Input layer
model.add(Dense(156, input_shape=(num_inputs,)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Hidden layer
model.add(Dense(156))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Output layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# %%
SVG(model_to_dot(model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))

# %%
model.fit(X, y, batch_size=64, epochs=50, validation_split=0.20, callbacks=[metrics])

# %%
# Plot the cost against epoch
f1s_plot, = plt.plot(metrics.val_epochs, metrics.val_f1s, 'r+')
precisions_plot, = plt.plot(metrics.val_epochs, metrics.val_precisions, 'b*')
recalls_plot, = plt.plot(metrics.val_epochs, metrics.val_recalls, 'g^')

plt.legend([f1s_plot, precisions_plot, recalls_plot], ["F1-Score", "Precision", "Recall" ])

# %%

# data directory and csv file should have the same name
DATA_PATH = 'test'

test = pd.read_csv('./data/' + DATA_PATH + '.csv')

features_test_file = Path("./features_test.pkl")

if not features_test_file.is_file():
    features = test.apply(extract_features, axis=1)
    dump_features(features, features_test_file)
    test = test.assign(features=features.values)
else:
    features = pd.read_pickle('./features_test.pkl')
    test = test.assign(features=features.values)

# %%
X_test = np.array(test.loc[:, 'features'])
X_test = np.vstack(X_test)

X_test.shape

# calculate predictions
predictions = model.predict_classes(X_test)
predict_class = lb.inverse_transform(predictions)

test['Class'] = predict_class
test_output = test.copy()

# drop the 'feature' column as it is not required for submission
test_output = test_output.drop(columns=['features'], axis=1)

test_output.to_csv('sub01.csv', index=False)
