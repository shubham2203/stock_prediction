import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import glob
import datetime
import streamlit as st
# %matplotlib inline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

st.title('Stock Prediction Website')

st.subheader('FA LSTM MODEL 1')

option = st.selectbox(
    'Select The Stock ',
    ('Marico','Axis', 'ICICI', 'INFY'))

if option=='Marico':
	df= pd.read_excel('marico.xlsx')
	ds=pd.read_csv('MARICO.NS.csv')
	df.head()
    

if option=='Axis':
	df= pd.read_excel('axis.xlsx')
	ds=pd.read_csv('AXISBANK.NS.csv')
	df.head()
   

if option=='ICICI':
	df= pd.read_excel('icici.xlsx')
	ds=pd.read_csv('ICICIBANK.NS.csv')
	df.head()



if option=='INFY':
	df= pd.read_excel('infosys.xlsx')
	ds=pd.read_csv('INFY.NS.csv')
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
plt.title("FA LSTM 1 Prediction Graph")
plt.xlabel("2008 - 2022 All Quaters ")
plt.ylabel("tbd")
st.pyplot(fig)

from sklearn.metrics import mean_squared_error, r2_score
import math
print('Mean error: %.2f' % math.sqrt(mean_squared_error(closing_price, valid['Price'])))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(closing_price, valid['Price']))

###################################################### - FA LSTM Model 2 Code 

st.subheader('FA LSTM MODEL 2 ')

df.index = df.Time
df.drop('Time', axis=1, inplace=True)

#creating train and test sets
dst=df.values
train = dst[0:41,:]
valid = dst[41:51,:]
test=dst[51:,:]

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
model.compile(loss='mean_squared_error', optimizer='adam')

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
plt.title("FA LSTM 2 Prediction Graph")
plt.xlabel("2008 - 2022 All Quaters ")
plt.ylabel("tbd")
st.pyplot(fig)

from sklearn.metrics import mean_squared_error, r2_score
import math
print('Root Mean error: %.2f' % math.sqrt(mean_squared_error(lstmtestPred,testSeq_y)))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(lstmtestPred,testSeq_y))

###################################### LSTM Code

st.subheader('TA LSTM MODEL 1 ')

data = ds.sort_index(ascending=True, axis=0)
data.fillna(method="bfill",inplace=True) 
new_data = pd.DataFrame(index=range(0,len(ds)),columns=['Date', 'Close'])
for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

     data = ds.sort_index(ascending=True, axis=0)
data.fillna(method="bfill",inplace=True) 
new_data = pd.DataFrame(index=range(0,len(ds)),columns=['Date', 'Close'])
for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

 #setting index as date
ds['Date'] = pd.to_datetime(ds.Date,format='%d-%m-%Y')
ds.index = ds['Date']

 #plot
plt.figure(figsize=(16,8))
plt.plot(ds['Close'], label='Close Price history')

new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

 #creating train and test sets
dataset = new_data.values

train = dataset[0:2898,:]
valid = dataset[2898:,:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(90,len(train)):
     x_train.append(scaled_data[i-90:i,0])
     y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

 # create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=15, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=15))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1)


inputs = new_data[len(new_data) - len(valid) - 90:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(90,inputs.shape[0]):
     X_test.append(inputs[i-90:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train = new_data[:2898]
valid = new_data[2898:]
valid['Predictions'] = closing_price
fig=plt.figure(figsize=(16,8))
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.plot()  
st.pyplot(fig)


from sklearn.metrics import mean_squared_error, r2_score

print('Mean squared error: %.2f' % mean_squared_error(closing_price, valid['Close']))

 # The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(closing_price, valid['Close']))


########################## RF Model 1

# st.subheader('FA RANDOM FOREST 1')

# if option=='Marico':
    
#      X = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]].values
#      Y = df.iloc[:,[12]].values

# if option=='ICICI':
    
#      X = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]].values
#      Y = df.iloc[:,[12]].values

# if option=='Axis':
    
#      X = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]].values
#      Y = df.iloc[:,[13]].values

# if option=='INFY':
    
#      X = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]].values
#      Y = df.iloc[:,[14]].values


# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range=(0, 1))
# X_new = sc.fit_transform(X)
# Y_new=sc.fit_transform(Y)


# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test = train_test_split(X_new,Y_new,test_size=0.30,random_state=0)




# from sklearn.ensemble import RandomForestRegressor
# classifier = RandomForestRegressor(n_estimators=100,random_state=0)
# classifier.fit(X_train,Y_train)


#  # In[ ]:





# # # In[90]:


# y_pred = classifier.predict(X_test)


# # # In[91]:


# y_pred=y_pred.reshape(-1,1)


# # # In[92]:


# y_pred.shape


# # # In[93]:


# y_pred = sc.inverse_transform(y_pred)


# # # In[94]:


# Y_test=sc.inverse_transform(Y_test)
# Y_train=sc.inverse_transform(Y_train)


# # # In[95]:



# fig=plt.figure(figsize=(16,8))
# plt.plot(Y_test)
# plt.plot(y_pred)
# plt.plot()
# st.pyplot(fig)




# from sklearn.metrics import mean_squared_error, r2_score

# print('Mean squared error: %.2f' % mean_squared_error(Y_test, y_pred))

#  # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f' % r2_score(Y_test, y_pred))


###################################################### - Dynamic( Directly Form Yahoo) LSTM Code 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model

st.subheader('DYNAMIC LSTM ')
 
start = '2010-01-01'
end = '2019-12-31'

user_input = st.text_input('Enter Name Of The Stock','AAPL')
df= data.DataReader(user_input, 'stooq', start, end)

#describing data
st.subheader('Data from 2010-2019')
st.write(df.describe())

#Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100= df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100= df.Close.rolling(100).mean()
ma200= df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

# splitting data into training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#loading  the model
model = load_model('keras_model.h5')

#testing part
past_100_days = data_training['Close'].tail(100).values

# Extract the values of the data_testing dataframe as a numpy array
data_testing_array = data_testing.values

# Concatenate the past_100_days array and the data_testing_array using the pd.concat function
past_100_days_df = pd.DataFrame(past_100_days)

# Concatenate the past_100_days_df DataFrame and the data_testing DataFrame using the pd.concat function
final_df = pd.concat([past_100_days_df, data_testing], ignore_index=True)

# Extract only the Close column from the final_df dataframe
input_data = final_df['Close'].values

# Reshape the input data to have a shape of (batch_size, sequence_length, features)
input_data = input_data.reshape(input_data.shape[0], 1, 1)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor
#final graph

st.subheader('Prediction vs Original - TBD Feature')
fig2 = plt.figure(figsize=(12,6))
plt.plot(ma100, 'b', label = 'Original Price')
plt.plot(ma200, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

######################## - End Of Code 




