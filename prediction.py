# Prediction.py

import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools


def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}

    print('Extracting feature from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []

        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat,
                     nfilt=config.nfilt, nfft=config.nfft)
            x = (x - config.min) / (config.max - config.min)
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)

            y_hat = model.predict(x, batch_size=10, verbose=0)
        
        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat))
        y_true.append(c)

        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
    return y_true, y_pred, fn_prob


df = pd.read_csv('csv_file/df_test.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles', 'conv.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    
model = load_model(config.model_path)


y_true, y_pred, fn_prob = build_predictions('data_test')
acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

print('predict score:', acc_score)

# y_probs = []
# for i, row in df.iterrows():
#     y_prob = fn_prob[row.fname]
#     y_probs.append(y_prob)
#     for c, p in zip(classes, y_prob):
#         df.at[i, c] = p
        
# y_pred = [classes[np.argmax(y)] for y in y_probs]
# df['y_pred'] = y_pred

# df.to_csv('test_pre.csv', index=False)

cm = confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(cm, classesd,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classesd))
    plt.xticks(tick_marks, classesd, rotation=45)
    plt.yticks(tick_marks, classesd)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plot_confusion_matrix(cm, classes, title='Prediction 144 instances of testing data)')

plt.tight_layout()

plt.savefig('pred/Confusion_matrix17.png', dpi=600)
plt.show()




