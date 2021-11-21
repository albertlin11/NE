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
y = df.iloc[:,5:6]
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
#load model, train with different patience
MLP= tf.keras.models.load_model('models/MLP_logid_hetero_888.h5')  
my_callbacks= [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)]
history =MLP.fit(X_mlp_train,Y_train, epochs= 100000,batch_size = 64,validation_split = 0.2, callbacks=my_callbacks)
MLP.save('models/mlp_log_888_100.h5')

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
mlp_tot_real = np.power(10,mlp_pred_inv)
rmse_mlp_tot = mean_squared_error(y, mlp_pred_inv, squared=False)
R2_mlp_tot = r2_score(y, mlp_pred_inv)
y_real =  np.power(10,y)
y_tot = list(y_real.reshape(shape))
y_tot_pred = list(mlp_tot_real.reshape(shape))
y_mlp_test = list(y_test_inv.reshape(shape_test))
y_mlp_test_pred = list(mlp_test_pred_inv.reshape(shape_test))


save_data_mlp = [y_mlp_test,y_mlp_test_pred,y_tot,y_tot_pred]
columns = ['y_mlp_test','y_mlp_test_pred','y_tot','y_tot_pred']
df_test_mlp= DataFrame(save_data_mlp,index=columns)
df_test_mlp = df_test_mlp.T
df_test_mlp.to_csv('prediction/MLP_logid_hetero_888_100.csv',index=True)
