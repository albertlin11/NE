# -*- coding: utf-8 -*-


#normalization
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
w = df.iloc[:,0:1] #width
l = df.iloc[:,1:2] #length
g = df.iloc[:,2:3] #It is actually Vd in Die11-idvg.csv
d = df.iloc[:,3:4] #It is Vg in Die11-idvg.csv
y = df.iloc[:,5:6] 

w =np.array(w, dtype=np.float32)
l =np.array(l, dtype=np.float32)
g =np.array(g, dtype=np.float32)
d =np.array(d, dtype=np.float32)
y =np.array(y, dtype=np.float32)

scalerw = MinMaxScaler()
w_minmax =scalerw.fit_transform(w)
scalerl = MinMaxScaler()
l_minmax =scalerl.fit_transform(l)
scalerg = MinMaxScaler()
g_minmax =scalerg.fit_transform(g)
scalerd = MinMaxScaler()
d_minmax =scalerd.fit_transform(d)
scalery = MinMaxScaler()
y_minmax =scalery.fit_transform(y)

rand = np.random.permutation(y.shape[0])

Y = y_minmax[rand]
L = l_minmax[rand]
W = w_minmax[rand]
G = g_minmax[rand]
D = d_minmax[rand]


split_percentage = 0.9
L_train = L[:int(l.shape[0]*split_percentage)]
L_test = L[int(l.shape[0]*split_percentage):]
W_train = W[:int(l.shape[0]*split_percentage)]
W_test = W[int(l.shape[0]*split_percentage):]
G_train = G[:int(l.shape[0]*split_percentage)]
G_test = G[int(l.shape[0]*split_percentage):]
D_train = D[:int(l.shape[0]*split_percentage)]
D_test = D[int(l.shape[0]*split_percentage):]
X_train = [W_train,L_train,G_train,D_train]
X_test = [W_test,L_test,G_test,D_test]   
Y_train = Y[:int(l.shape[0]*split_percentage)]
Y_test = Y[int(l.shape[0]*split_percentage):]



#%%



#load model, train with different patience
best_model= tf.keras.models.load_model('models/GA_logid_hetero_546.h5')  
my_callbacks= [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)]
history =best_model.fit(X_train,Y_train, epochs= 100000,batch_size = 64,validation_split = 0.2, callbacks=my_callbacks)
best_model.save('models/GA_logid_hetero_546_100.h5')
val_loss = history.history['val_loss']
best_model.summary()
#predict test dataset
ga_best_test = best_model.predict(X_test)
shape_test = Y_test.shape[0]
ga_best_test=ga_best_test.reshape((shape_test,1))
ga_best_test_inv = scalery.inverse_transform(ga_best_test)
ga_best_test_real = np.power(10,ga_best_test_inv)

y_test_inv = scalery.inverse_transform(Y_test)
y_test_real = np.power(10,y_test_inv)
rmse_ga_test = mean_squared_error(y_test_inv, ga_best_test_inv, squared=False)
R2_ga_test = r2_score(y_test_inv,ga_best_test_inv)

#predict train dataset
ga_best_train =best_model.predict(X_train)
shape_train = Y_train.shape[0]
ga_best_train=ga_best_train.reshape((shape_train,1))
ga_best_train_inv = scalery.inverse_transform(ga_best_train)
ga_best_train_real = np.power(10,ga_best_train_inv)
y_train_inv = scalery.inverse_transform(Y_train)
y_test_real = np.power(10,y_train_inv)
rmse_ga_train = mean_squared_error(y_train_inv, ga_best_train_inv, squared=False)
R2_ga_train = r2_score(y_train_inv, ga_best_train_inv)

#predict total dataset
ga_best_tot =best_model.predict([w_minmax,l_minmax,g_minmax,d_minmax])
shape = y.shape[0]
ga_best_tot=ga_best_tot.reshape((shape,1))
ga_best_tot_inv = scalery.inverse_transform(ga_best_tot)
ga_best_tot_real = np.power(10,ga_best_tot_inv)
y_real =  np.power(10,y)
rmse_ga_tot = mean_squared_error(y, ga_best_tot_inv, squared=False)
R2_ga_tot = r2_score(y, ga_best_tot_inv) 

y_test = list(y_test_inv.reshape(shape_test))
y_test_pred = list(ga_best_test_inv.reshape(shape_test))
y_tot = list(y_real.reshape(shape))
y_tot_pred = list(ga_best_tot_real.reshape(shape))

save_data = [y_test,y_test_pred,y_tot,y_tot_pred]
columns = ['y_test','y_test_pred','y_tot','y_tot_pred']
df_test= DataFrame(save_data,index=columns)
df_test = df_test.T
df_test.to_csv("prediction/GA_logid_hetero_546_100.csv",index=True)