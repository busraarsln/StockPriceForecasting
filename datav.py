#import package
import pandas as pd
from pandas_datareader import data
import numpy as np
import datetime
from datetime import date
from workalendar.europe import Turkey
#to plot within notebook
import matplotlib.pyplot as plt
import csv
import math
import sys
from workalendar.europe import Turkey
from keras.models import load_model
#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
#for normalizing data
from sklearn.preprocessing import MinMaxScaler

#importing required libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from math import sqrt

sc= MinMaxScaler(feature_range=(0,1))

df = data.DataReader('AKBNK.IS','yahoo')
df['Date'] = df.index
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')

data = df.sort_index(ascending=True, axis=0)
data['Average']=(data['High']+data['Low'])/2
input_feature=pd.DataFrame(index=range(0,len(df)),columns=['Date','Average','Volume','Open','Close'])
for i in range(0,len(data)):
    input_feature['Date'][i]=data['Date'][i]
    input_feature['Average'][i]=data['Average'][i]
    input_feature['Volume'][i]=data['Volume'][i]
    input_feature['Open'][i]=data['Open'][i]
    input_feature['Close'][i]=data['Close'][i]
input_feature.index=input_feature.Date
input_feature.drop('Date',axis=1,inplace=True)


lookback=50
test_size=int(.7*len(data))


input_data=input_feature
veri=input_data.values
dataset=pd.DataFrame(veri)
train = dataset.iloc[0:test_size,:]
valid = dataset.iloc[test_size: ,:]

fit_data=np.zeros(shape=(train.shape[0],train.shape[1]))
col=np.array([])
for i in range(train.shape[1]):
    col=train.iloc[:,i]
    seri=pd.DataFrame(col)
    fit_data=np.column_stack((fit_data,sc.fit_transform(seri)))
fit_data=np.delete(fit_data,np.s_[0:4:],axis=1)

x_train,y_train=[],[]

for i in range(lookback,len(train)):
    x_train.append(fit_data[i-lookback:i,])
    y_train.append(fit_data[i,3])

x_train, y_train = np.array(x_train), np.array(y_train)

model=load_model('LSTM_AKBANK_Multi2.h5')
#model = Sequential()
#model.add(LSTM(units=30, return_sequences= True, input_shape=(x_train.shape[1],4)))
#model.add(LSTM(units=30, return_sequences=True))
#model.add(LSTM(units=30))
#model.add(Dense(units=1))
#model.summary()
#
#model.compile(optimizer='adam', loss='mean_squared_error')
#
#model.fit(x_train, y_train, epochs=50, batch_size=32)
#
#file_name = 'LSTM_AKBANK_Multi2.h5'
#model.save(file_name)
#print("Saved model `{}` to disk".format(file_name))


valid_inputs = input_data[len(input_data) - len(valid) - lookback:]

fit_data_valid=np.zeros(shape=(valid_inputs.shape[0],valid_inputs.shape[1]))
col2=np.array([])
for i in range(valid.shape[1]):
    col2=valid_inputs.iloc[:,i]
    seri_valid=pd.DataFrame(col2)
    fit_data_valid=np.column_stack((fit_data_valid,sc.fit_transform(seri_valid)))
fit_data_valid=np.delete(fit_data_valid,np.s_[0:4:],axis=1)




busra=valid.iloc[:,3]
data_bla=np.array(busra)
data_bla = np.reshape(data_bla, (data_bla.shape[0],1))

X_test = []
for i in range(50,fit_data_valid.shape[0]):
    X_test.append(fit_data_valid[i-lookback:i,])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],4))
closing_price = model.predict(X_test)
closing_price = sc.inverse_transform(closing_price)
#
rms=np.sqrt(np.mean(np.power((data_bla-closing_price),2)))
print(rms)


train = data[:test_size]
valid = data[test_size:]
valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])

an = datetime.datetime.now()
tarih=datetime.datetime.strftime(an, '%X')

if tarih < '18:05:00':
    last_date=df['Date'].iloc[-2]
    last_fifth_days=input_feature[-51:-1]
else:
    last_date=df['Date'].iloc[-1]
    last_fifth_days=input_feature[-50:]
    
print(last_date)
next_date= last_date + datetime.timedelta(days=1)
cal=Turkey()
#
def control_date(ndate):
    result=cal.is_working_day(date(next_date.year,next_date.month,next_date.day))
    return result
while control_date(next_date)==False:
    next_date= next_date + datetime.timedelta(days=1)
    control_date(next_date)

print(next_date)
#
#ten_days=sc.transform(last_fifth_days)

fifth_days=np.zeros(shape=(last_fifth_days.shape[0],last_fifth_days.shape[1]))
col3=np.array([])
for i in range(last_fifth_days.shape[1]):
    col3=last_fifth_days.iloc[:,i]
    fifth_valid=pd.DataFrame(col3)
    fifth_days=np.column_stack((fifth_days,sc.fit_transform(fifth_valid)))
fifth_days=np.delete(fifth_days,np.s_[0:4:],axis=1)


#close_value=last_fifth_days.iloc[3]
#data_close=np.array(close_value)
#data_close = np.reshape(data_close, (data_close.shape[0],1))

fifth_days = np.reshape(fifth_days,(1,len(fifth_days),4))
predicted_date=model.predict(fifth_days)
print(predicted_date)
predicted_date = sc.inverse_transform(predicted_date)
print(predicted_date)
sonuc={'Date':[],
       'Prediction':[]}
sonuc=pd.DataFrame(sonuc)
sonuc.to_csv (r'C:\Users\busra\Desktop\Project\sonuc.csv', index = None, header=True) 

#
sonuc=sonuc.append([{'Date':next_date,'Prediction':predicted_date}])
with open('sonuc.csv','a') as newFile:
    newFileWriter = csv.writer(newFile)
#    newFileWriter.writerow(['Date','Prediction','Price'])
    newFileWriter.writerow([next_date,predicted_date])
