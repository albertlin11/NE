# -*- coding: utf-8 -*-


import tensorflow as tf
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from pandas.core.frame import DataFrame
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

epochs=[]
Rs=[]
for i in range(30):
    seed_value=i+1
    
    os.environ['PYTHONASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)  
    df = pd.read_csv('Die11-idvg-tot.csv')
    column = [0,1,2,3,4,5,6,7]
    df.columns= column
    
    #normalization
    x = df.iloc[:,0:4]
    x =np.array(x, dtype=np.float32)
    
    # y = df.iloc[:,4:5] # Linear Id
    y = df.iloc[:,5:6] # Log Id
    # y = df.iloc[:,7:8] # Cornell Id
    y =np.array(y, dtype=np.float32)
    
    
    scalerx = MinMaxScaler()
    x_minmax =scalerx.fit_transform(x)
    scalery = MinMaxScaler()
    y_minmax =scalery.fit_transform(y)
    
    rand = np.random.permutation(y.shape[0])
    
    X = x_minmax[rand]
    Y = y_minmax[rand]
    
    split_percentage = 0.9
    X_mlp_train = X[:int(x.shape[0]*split_percentage)]
    X_mlp_test = X[int(x.shape[0]*split_percentage):]
    Y_train = Y[:int(x.shape[0]*split_percentage)]
    Y_test = Y[int(x.shape[0]*split_percentage):]
    
    #MLP model
    my_callbacks1= [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    MLP= tf.keras.Sequential()
    MLP.add(tf.keras.layers.Dense(14, activation="sigmoid"))
    MLP.add(tf.keras.layers.Dense(14, activation="sigmoid"))
    MLP.add(tf.keras.layers.Dense(14, activation="sigmoid"))
    MLP.add(tf.keras.layers.Dense(14, activation="sigmoid"))
    MLP.add(tf.keras.layers.Dense(1))  
    MLP.compile(optimizer =opt, loss = 'mean_squared_error', metrics = ['mse'])
    history_mlp1 = MLP.fit(X_mlp_train,Y_train, epochs= 5000,batch_size = 64,validation_split = 0.2, callbacks=my_callbacks1)
    loss=history_mlp1.history['loss']
    epoch = len(loss)
    # MLP.summary()
    #MLP.save('Models/randommlp10-3'+str(i+1)+'test.h5')
    
    
    #predict test dataset
    mlp_test_pred =MLP.predict(X_mlp_test)
    shape_test = Y_test.shape[0]
    mlp_test_pred=mlp_test_pred.reshape((shape_test,1))
    mlp_test_pred_inv = scalery.inverse_transform(mlp_test_pred)
    
    y_test_inv = scalery.inverse_transform(Y_test)
    rmse_mlp_test = mean_squared_error(y_test_inv, mlp_test_pred_inv, squared=False)
    R2_mlp_test = r2_score(y_test_inv, mlp_test_pred_inv)


    #predict train dataset
    mlp_train_pred =MLP.predict(X_mlp_train)
    shape_train = Y_train.shape[0]
    mlp_train_pred=mlp_train_pred.reshape((shape_train,1))
    mlp_train_pred_inv = scalery.inverse_transform(mlp_train_pred)
    
    y_train_inv = scalery.inverse_transform(Y_train)
    rmse_mlp_train = mean_squared_error(y_train_inv, mlp_train_pred_inv, squared=False)
    R2_mlp_train = r2_score(y_train_inv, mlp_train_pred_inv)

    #predict total dataset
    mlp_pred = MLP.predict(x_minmax)
    shape = y.shape[0]
    mlp_pred=mlp_pred.reshape((shape,1))
    mlp_pred_inv = scalery.inverse_transform(mlp_pred)
    rmse_mlp_tot = mean_squared_error(y, mlp_pred_inv, squared=False)
    R2_mlp_tot = r2_score(y, mlp_pred_inv)

    R = [rmse_mlp_test,R2_mlp_test,rmse_mlp_train,R2_mlp_train,rmse_mlp_tot,R2_mlp_tot]

    Rs.append(R)
    epochs.append(epoch)
    
# column_output=['rmse_mlp_test','R2_mlp_test','rmse_mlp_train','R2_mlp_train','rmse_mlp_tot','R2_mlp_tot']
df_random= DataFrame(Rs)
# df_random = df_random.T
df_random.to_csv("Hetero/random_MLP_logid_hetero_14-4_100.csv",index=True)
    