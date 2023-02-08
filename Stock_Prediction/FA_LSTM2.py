


import pandas as pd
df= pd.read_excel('marico.xlsx')
df.head()

import keras
import tensorflow as tf

import random as rn

import numpy as np
import pandas as pd
import os
import glob
import datetime
import streamlit as st

st.title('Stock Prediction Website')

df.index = df.Time
df.drop('Time', axis=1, inplace=True)

#creating train and test sets
ds=df.values
train = ds[0:41,:]
valid = ds[41:51,:]
test=ds[51:,:]

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
trainMinmax = min_max_scaler.fit_transform(train) #fit and transform training data
valMinmax = min_max_scaler.transform(valid)
testMinmax = min_max_scaler.transform(test)

from numpy import array

def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, AveragePooling1D,MaxPooling1D
from keras.layers import Conv1D,AveragePooling1D,MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD, Nadam

#from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1,l2,l1_l2

EarlyStop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto',restore_best_weights=True)
epochs = 70
lr =0

sgd = SGD(lr=lr) 
bs = 10

n_steps_in =3
n_steps_out = 1


trainSeq_x, trainSeq_y = split_sequences(trainMinmax, n_steps_in,n_steps_out)

validationSeq_x, validationSeq_y= split_sequences(valMinmax, n_steps_in,n_steps_out)

testSeq_x, testSeq_y= split_sequences(testMinmax, n_steps_in,n_steps_out)

X_useless, y_useless = split_sequences(trainMinmax, n_steps_in,n_steps_out)
n_features = X_useless.shape[2]

np.random.seed(0); print(np.random.rand(4))

model = Sequential()
model.add(LSTM(300, 
               input_shape=(n_steps_in, n_features),
               activation = 'tanh')) 
model.add(Dropout(0.1))
model.add(Dense(1,activation = 'linear'))
model.compile(loss='mean_squared_error', optimizer='sgd')

#model
model.fit(trainSeq_x, trainSeq_y,batch_size=bs,epochs=epochs, callbacks= [EarlyStop] ,verbose=2, shuffle=False,
                         validation_data =(validationSeq_x, validationSeq_y))

# Commented out IPython magic to ensure Python compatibility.

lstmValPred = model.predict(validationSeq_x)

lstmtestPred = model.predict(testSeq_x)

model.summary()


import matplotlib.pyplot as plt
# %matplotlib inline

fig=plt.figure(figsize=(16,8))

plt.plot(trainSeq_y)
plt.plot(testSeq_y,color="red")
plt.plot(lstmtestPred,color="green")
plt.plot()
st.pyplot(fig)

from sklearn.metrics import mean_squared_error, r2_score
import math
print('Root Mean error: %.2f' % math.sqrt(mean_squared_error(lstmtestPred,testSeq_y)))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(lstmtestPred,testSeq_y))