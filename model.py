import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Activation
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical, plot_model
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from cfg import Config
import time

config = Config(mode ='conv')


def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None


def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    
    x = []
    y = []

    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label == rand_class].index)
        rate, wav = wavfile.read('clean_data/' + file)
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0] - config.step)
        sample = wav[rand_index: rand_index + config.step]
        x_sample = mfcc(sample, rate, numcep=config.nfeat, 
                        nfilt=config.nfilt, nfft=config.nfft)

        _min = min(np.amin(x_sample), _min)
        _max = max(np.amax(x_sample), _max)
        x.append(x_sample)
        y.append(classes.index(label))
    #     print(classes.index(label))
    config.min = _min
    config.max = _max
    x, y = np.array(x), np.array(y)
    # y = np.array(y)
    x = (x - _min) / (_max - _min)

    if config.mode == 'conv':
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    elif config.mode == 'time':
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    y = to_categorical(y, num_classes=3)
    # print(y)
    config.data = (x, y)

    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)

    return x, y


def get_conv_model():
    model = Sequential()
    # model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1),
    #                  padding='same', input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
    #                  padding='same'))
    # model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1),
    #                  padding='same'))
    # model.add(MaxPooling2D((2,2)))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(3, activation='softmax'))
    # model.summary()
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['acc'])
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    # opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    model.summary()
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    
    return model


df = pd.read_csv('csv_file/df_train.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean_data/' + f)
    df.at[f, 'length'] = signal.shape[0] / rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2 * int(df['length'].sum() / 0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)


if config.mode == 'conv':
    x, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (x.shape[1], x.shape[2], 1)
    model = get_conv_model()


class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)

history = model.fit(x, y, epochs=300,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.2,
                    callbacks=[checkpoint],
                    class_weight=class_weight
                    )

model.save(config.model_path)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.tight_layout()
# plt.savefig('fig/acc14.png', dpi=600)
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.tight_layout()
# plt.savefig('fig/loss14.png', dpi=600)
plt.show()


