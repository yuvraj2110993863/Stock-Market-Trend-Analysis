import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from datetime import date
import yfinance as yf
import streamlit as st
from keras.models import load_model

start="2010-01-01"
end=str(date.today())



st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker','AAPL')
#df=data.DataReader(user_input,'yahoo',start,end)
df=yf.download(user_input, start, end)[['Adj Close','Open', 'High', 'Low', 'Close', 'Volume']]

st.subheader('Data')
st.write(df.describe())

#Visualisations
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 Moving Averages ')
df=df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])

# Sort the DataFrame by date
df = df.sort_values('Date')



df['100_MA'] = df['Close'].rolling(window=100).mean()
fig=plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Stock Closing Prices')
plt.plot(df['Date'], df['100_MA'], label='100-Day Moving Average', color='red')
plt.title('Stock Market Closing Prices and 100-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)

df['Date'] = pd.to_datetime(df['Date'])


st.subheader('Closing Price vs Time Chart with 100 and 200 Moving Averages ')
df['100_MA'] = df['Close'].rolling(window=100).mean()
df['200_MA'] = df['Close'].rolling(window=200).mean()
fig=plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Stock Closing Prices')
plt.plot(df['Date'], df['100_MA'], label='100-Day Moving Average', color='red')
plt.plot(df['Date'], df['200_MA'], label='200-Day Moving Average', color='green')
plt.title('Stock Market Closing Prices with 100-Day and 200-Day Moving Averages')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
st.pyplot(fig)

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)
x_train=[]
y_train=[]
for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)

model=load_model('keras_model.h5')
past_100_days=data_training.tail(100)
final_df=pd.concat([past_100_days,data_testing],ignore_index=True)
input_data=scaler.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

fig=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig)