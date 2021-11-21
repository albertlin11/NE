# -*- coding: utf-8 -*-


import tensorflow as tf
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
seed_value = 1
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
y = df.iloc[:,5:6] # Log Id
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

#%%
my_callbacks1= [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
mlp= tf.keras.Sequential()
mlp.add(tf.keras.layers.Dense(8, activation="sigmoid"))
mlp.add(tf.keras.layers.Dense(8, activation="sigmoid"))
mlp.add(tf.keras.layers.Dense(8, activation="sigmoid"))
# mlp.add(tf.keras.layers.Dense(7, activation="sigmoid"))
# mlp.add(tf.keras.layers.Dense(10, activation="sigmoid"))
mlp.add(tf.keras.layers.Dense(1))  
mlp.compile(optimizer =opt, loss = 'mean_squared_error', metrics = ['mse'])
history_mlp1 = mlp.fit(X_mlp_train,Y_train, epochs= 1000,batch_size = 64,validation_split = 0.2, callbacks=my_callbacks1)
mlp.summary()
mlp.save('models/MLP_logid_hetero_888.h5')

