#!/usr/bin/env python
import json
from datetime import datetime, timedelta
import sys
import csv
import glob
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics
import os



id_1 = '2330'
id_2 = "3008"

input_path= sys.argv[1]
out_path   = sys.argv[2]


filelist = glob.glob(os.path.join(out_path, "/*.json"))
print("filelist=",filelist)
for f in filelist:
    
    os.remove(f)




i = 1
pred_final =[]
path = input_path + "/*.csv"
print("input_pathy=", path)
files=glob.glob(path)  
print(files)
for file in files: 
    print("file")
    df = pd.read_csv(file)
    L = len(df)
    t = np.array([df.ix[:, 2]])
    t = t[:, 0:L-1]
    #print(df.head(5))
    df = df.drop(df.index[[0]])
    df.reset_index(inplace=True)
    #print(df.head(5))
    #print(len(df))
    #print(len(t))
    t= pd.DataFrame(t)
    #print(t)
    df2 = t.T

    concat = pd.concat([df,df2],  axis=1)
    #print(concat.columns)
    concat.rename(index=str, columns={0: "next_date_price"},inplace=True)
    #print(concat)

    concat = concat.iloc[:, 3:]
    #print(concat)
    concat = concat.iloc[::-1]
    train = concat.iloc[:2*L/3, :]
    X_train = train.iloc[:, :4]
    Y_train = train.iloc[:, 4]



    test = concat.iloc[2*L/3:, :]
    X_test = test.iloc[:, :4]
    Y_test = test.iloc[:, 4]



    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_train= np.reshape(X_train, (X_train.shape[0],1,X_test.shape[1]))
    #print("Xtrain=", X_train.shape)

    #print("Y_train=",Y_train)
    #print(Y_train.shape)
    Y_train = Y_train.values.reshape(-1,1)
    #print(Y_train.shape)
    scaler1 = MinMaxScaler()
    scaler1.fit(Y_train)
    Y_train = scaler1.transform(Y_train)

    #print("train", train)
    #print("len(X_test)", len(X_test))
    new_row = [260.00,260.00,257.00,25228536]
    X_test = np.vstack([X_test, new_row])
    #print("len(X_test)", len(X_test))


    scaler2 = MinMaxScaler()
    scaler2.fit(X_test)
    X_test = scaler2.transform(X_test)
    X_test= np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))

    Y_test = Y_test.values.reshape(-1,1)
    scaler3 = MinMaxScaler()
    scaler3.fit(Y_test)
    Y_test = scaler3.transform(Y_test)

    model = Sequential()

    model.add(LSTM(20,activation = 'tanh',input_shape = (1,4),recurrent_activation= 'hard_sigmoid'))
 
    model.add(Dense(1))

    model.compile(loss= 'mean_squared_error',optimizer = 'rmsprop', metrics=[metrics.mae])

    model.fit(X_train,Y_train,epochs=20,verbose=2, batch_size=30, validation_split=0.2)

#scores = model.evaluate(X_test, Y_test, verbose=1)
#print(scores[1])
#df2 = pd.DataFrame({ 'close':260.00, 'high': 260.00, 'low': 257.00, 'volume': 25228536 })
#X_test = pd.concat([X_test,df2])
#print(X_test)
#print(X_test.shape)

#print("Xtest=", X_test.tail())
    Predict = model.predict(X_test)
    #print("predict=",Predict)
    pre_today = scaler3.inverse_transform(Predict)[-1]
    act_yes = scaler3.inverse_transform(Y_test)[-1]
    #print("pre_today", type(pre_today[0]))
    a = {}
    if i ==1:
        a.update({'id': id_1})
        if pre_today - act_yes > 0:
            a.update({"type": "buy"})
            a.update({"weight": 1})
            a.update({"open_price": act_yes.tolist()[0] })
            a.update({"close_high_price": pre_today.tolist()[0] })
            a.update({"life": 5})
            
            
        else:
            a.update({"type": "short"})
            a.update({"weight": 1})
            a.update({"open_price": pre_today.tolist()[0] })
            a.update({"close_high_price": (pre_today.tolist()[0]-5) })
            a.update({"life": 5})
            
        pred_final.append(a)
        print("a=",pred_final)
    
    else:
        a.update({'id': id_2})
        if pre_today - act_yes > 0:
            a.update({"type": "buy"})
            a.update({"weight": 2})
            a.update({"open_price": act_yes.tolist()[0]})
            a.update({"close_high_price": pre_today.tolist()[0]})
            a.update({"life": 5})
            
            
        else:
            a.update({"type": "short"})
            a.update({"weight": 2})
            a.update({"open_price": pre_today.tolist()[0]})
            a.update({"close_high_price": (pre_today.tolist()[0]-5)})
            a.update({"life": 5})
        
        pred_final.append(a)
        print("a====", pred_final)
        
    i += 1
    
try:  
    os.mkdir(out_path)
except OSError:  
    print ("Creation of the directory %s failed" % out_path)
else:  
    print ("Successfully created the directory %s " % out_path)

f = open(out_path + '/result.json', 'w')
f.close()
with open(out_path + '/result.json', 'w') as fp:
    json.dump(pred_final, fp)


        