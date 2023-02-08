

import pandas as pd
ds= pd.read_excel('marico.xlsx')

# Commented out IPython magic to ensure Python compatibility.

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


import streamlit as st

st.title('Stock Prediction Website')
df=ds
df.head()

new_data = pd.DataFrame(index=range(0,len(df)),columns=['Time', 'Price'])
for i in range(0,len(df)):
    new_data['Time'][i] = df['Time'][i]
    new_data['Price'][i] = df['Price'][i]

new_data.index = new_data.Time
new_data.drop('Time', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:45,:]
valid = dataset[45:,:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(3,len(train)):
    x_train.append(scaled_data[i-3:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=10, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=10))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=3, verbose=2)

#predicting 57 values, using past 3 from the train data
inputs = new_data[len(new_data) - len(valid) - 3:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(3,inputs.shape[0]):
    X_test.append(inputs[i-3:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train = new_data[:45]
valid = new_data[45:]
valid['Predictions'] = closing_price
fig=plt.figure(figsize=(16,8))
plt.plot(train['Price'])
plt.plot(valid[['Price','Predictions']])
plt.plot()
st.pyplot(fig)

from sklearn.metrics import mean_squared_error, r2_score
import math
print('Mean error: %.2f' % math.sqrt(mean_squared_error(closing_price, valid['Price'])))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(closing_price, valid['Price']))