#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import time
import datetime
import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams ['font.sans-serif'] ='SimHei'               #显示中文
plt.rcParams ['axes.unicode_minus']=False               #显示负号
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score
 
import tensorflow as tf
import keras
from keras.layers import Layer
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import GRU, Dense,Conv1D, MaxPooling1D,GlobalMaxPooling1D,Embedding,Dropout,Flatten,SimpleRNN,LSTM,Bidirectional
import tensorflow as tf
from keras.callbacks import EarlyStopping
#from tensorflow.keras import regularizers
#from keras.utils.np_utils import to_categorical
from tensorflow.keras  import optimizers


# ### 读取数据

# In[2]:


data0=pd.read_csv('沪深300期货历史数据 (2).csv',parse_dates=['日期']).set_index('日期')[['开盘','高','低','收盘']].sort_index()
data0=data0.astype('float')
data0.head()


# ### 构建训练集和测试集

# In[3]:


def build_sequences(text, window_size=24):
    #text:list of capacity
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+window_size]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y)
 
def get_traintest(data,train_ratio=0.8,window_size=24):
    train_size=int(len(data)*train_ratio)
    train=data[:train_size]
    test=data[train_size-window_size:]
    X_train,y_train=build_sequences(train,window_size=window_size)
    X_test,y_test=build_sequences(test,window_size=window_size)
    return X_train,y_train[:,-1],X_test,y_test[:,-1]


# In[4]:


data=data0.to_numpy()
scaler = MinMaxScaler() 
scaler = scaler.fit(data[:,:-1])
X=scaler.transform(data[:,:-1])   
 
y_scaler = MinMaxScaler() 
y_scaler = y_scaler.fit(data[:,-1].reshape(-1,1))
y=y_scaler.transform(data[:,-1].reshape(-1,1))


# In[5]:


train_ratio=0.8     #训练集比例   
window_size=5      #滑动窗口大小，即循环神经网络的时间步长
X_train,y_train,X_test,y_test=get_traintest(np.c_[X,y],window_size=window_size,train_ratio=train_ratio)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# In[6]:


y_test1 = y_scaler.inverse_transform(y_test.reshape(-1,1))
test_size=int(len(data)*(1-train_ratio))
plt.figure(figsize=(10,5),dpi=256)
plt.plot(data0.index[:-test_size],data0.iloc[:,-1].iloc[:-test_size],label='训练集',color='#FA9905')
plt.plot(data0.index[-test_size:],data0.iloc[:,-1].iloc[-(test_size):],label='测试集',color='#FB8498',linestyle='dashed')
plt.legend()
plt.ylabel('沪深300',fontsize=14)
plt.xlabel('日期',fontsize=14)
plt.show()


# In[7]:


print(f'训练集开始时间{data0.index[:-test_size][0]},结束时间{data0.index[:-test_size][-1]}')
print(f'测试集开始时间{data0.index[-test_size:][0]},结束时间{data0.index[-test_size:][-1]}')


# ### 定义评价指标

# In[8]:


def set_my_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1)
    rn.seed(12345)
    tf.random.set_seed(123)
    
def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    mape=(abs(y_predict -y_test)/ y_test).mean()
    #r_2=r2_score(y_test, y_predict)
    return mse, rmse, mae, mape #r_2


# In[9]:


def build_model(X_train,mode='LSTM',hidden_dim=[32,16]):
    set_my_seed()
    model = Sequential()
    if mode=='MLP':
        model.add(Dense(hidden_dim[0],activation='relu',input_shape=(X_train.shape[-2],X_train.shape[-1])))
        model.add(Flatten())
        model.add(Dense(hidden_dim[1],activation='relu'))
    elif mode=='LSTM':
        # LSTM
        model.add(LSTM(hidden_dim[0],return_sequences=True, input_shape=(X_train.shape[-2],X_train.shape[-1])))#
        model.add(Dropout(0.4))
        model.add(LSTM(hidden_dim[1]))
        model.add(Dropout(0.5))
        #model.add(Flatten())
        #model.add(Dense(hidden_dim[1], activation='relu'))
    elif mode=='GRU':
        #GRU
        model.add(GRU(hidden_dim[0],return_sequences=True, input_shape=(X_train.shape[-2],X_train.shape[-1])))
        model.add(GRU(hidden_dim[1]))
    elif mode=='BiLSTM':
        # Bidirectional LSTM
        model.add(Bidirectional(LSTM(hidden_dim[0], return_sequences=True), input_shape=(X_train.shape[-2], X_train.shape[-1])))
        model.add(Dropout(0.4))
        model.add(Bidirectional(LSTM(hidden_dim[1])))
        model.add(Dropout(0.5))

    elif mode=='CNN+LSTM':
        # CNN followed by LSTM
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[-2], X_train.shape[-1])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(hidden_dim[0]))
        model.add(Dropout(0.4))
        model.add(Dense(hidden_dim[1], activation='relu'))

    elif mode=='CNN+BiLSTM':
        # CNN followed by BiLSTM
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[-2], X_train.shape[-1])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Bidirectional(LSTM(hidden_dim[0])))
        model.add(Dropout(0.4))
        model.add(Dense(hidden_dim[1], activation='relu'))    
    model.add(Dense(1))
    model.compile(optimizer='Adam', loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError(),"mape","mae"])
    return model


# In[10]:


def plot_loss(hist,imfname=''):
    plt.subplots(1,4,figsize=(16,4))
    for i,key in enumerate(hist.history.keys()):
        n=int(str('24')+str(i+1))
        plt.subplot(n)
        plt.plot(hist.history[key], 'k', label=f'Training {key}')
        plt.title(f'{imfname} Training {key}')
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
    plt.tight_layout()
    plt.show()
def plot_fit(y_test, y_pred):
    plt.figure(figsize=(4,2))
    plt.plot(y_test, color="red", label="actual")
    plt.plot(y_pred, color="blue", label="predict")
    plt.title(f"拟合值和真实值对比")
    plt.xlabel("Time")
    plt.ylabel('values')
    plt.legend()
    plt.show()


# In[11]:


df_eval_all=pd.DataFrame(columns=['MSE','RMSE','MAE','MAPE'])
df_preds_all=pd.DataFrame()
def train_fuc(mode='LSTM',batch_size=4,epochs=30,hidden_dim=[32,16],verbose=0,show_loss=True,show_fit=True):
    #构建模型
    s = time.time()
    set_my_seed()
    model=build_model(X_train=X_train,mode=mode,hidden_dim=hidden_dim)
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=5)
    hist=model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=[earlystop],validation_data=(X_test, y_test))  #
    print(hist.history.keys())
    if show_loss:
        plot_loss(hist)
            
    #预测
    y_pred = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    #print(f'真实y的形状：{y_test.shape},预测y的形状：{y_pred.shape}')
    if show_fit:
        plot_fit(y_test1, y_pred)
    e=time.time()
    print(f"运行时间为{round(e-s,3)}")
    df_preds_all[mode]=y_pred.reshape(-1,)
        
    s=list(evaluation(y_test1, y_pred))
    df_eval_all.loc[f'{mode}',:]=s
    s=[round(i,3) for i in s]
    print(f'{mode}的预测效果为：MSE:{s[0]},RMSE:{s[1]},MAE:{s[2]},MAPE:{s[3]}')
    print("=======================================运行结束==========================================")
    return hist


# In[42]:


window_size=5
batch_size=128
epochs=10
hidden_dim=[32,16]

verbose=0
show_fit=True
show_loss=True
mode='LSTM'  #MLP,GRU


# In[43]:


hist=train_fuc(mode='LSTM',batch_size=batch_size,epochs=epochs,hidden_dim=hidden_dim,verbose=1)


# In[44]:


loss=hist.history['loss']
val_loss=hist.history['val_loss']
# 绘制训练和验证的损失值
plt.figure(figsize=(7,3),dpi=128)
plt.plot(range(len(loss)), loss, 'p-', label='训练集损失')
plt.plot(range(len(loss)), val_loss, 'y--', label='测试集损失')
plt.title('损失')
plt.xlabel('训练轮数')
plt.ylabel('损失率')
plt.xticks(range(len(loss)))
plt.legend()
plt.show()


# In[45]:


#hist=train_fuc(mode='GRU',batch_size=batch_size,epochs=epochs,hidden_dim=hidden_dim,verbose=0)


# In[46]:


hist=train_fuc(mode='BiLSTM',batch_size=batch_size,epochs=epochs,hidden_dim=hidden_dim,verbose=0)


# In[47]:


hist=train_fuc(mode='CNN+LSTM',batch_size=batch_size,epochs=epochs,hidden_dim=hidden_dim,verbose=0)


# In[48]:


hist=train_fuc(mode='CNN+BiLSTM',batch_size=batch_size,epochs=epochs,hidden_dim=hidden_dim,verbose=0)


# In[ ]:





# In[ ]:





# ### 查看评价指标

# In[49]:


df_eval_all


# In[50]:


bar_width = 0.4
colors=['c', 'b', 'g', 'tomato', 'm', 'y', 'lime', 'k','orange','pink','grey','tan','gold','r']
fig, ax = plt.subplots(2,2,figsize=(10,8),dpi=128)
for i,col in enumerate(df_eval_all.columns):
    n=int(str('22')+str(i+1))
    plt.subplot(n)
    df_col=df_eval_all[col]
    m =np.arange(len(df_col))
    plt.bar(x=m,height=df_col.to_numpy(),width=bar_width,color=colors)
    
    #plt.xlabel('Methods',fontsize=12)
    names=df_col.index
    plt.xticks(range(len(df_col)),names,fontsize=10)
    plt.xticks(rotation=40)
    plt.ylabel(col,fontsize=14)
    
plt.tight_layout()
#plt.savefig('柱状图.jpg',dpi=512)
plt.show()


# In[51]:


plt.figure(figsize=(10,5),dpi=256)
for i,col in enumerate(df_preds_all.columns):
    plt.plot(data0.index[-test_size-1:],df_preds_all[col],label=col) # ,color=colors[i]

plt.plot(data0.index[-test_size-1:],y_test1.reshape(-1,),label='实际值',color='k',linestyle=':',lw=2)
plt.legend()
plt.ylabel('',fontsize=16)
plt.xlabel('日期',fontsize=14)
#plt.savefig('点估计线对比.jpg',dpi=256)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




