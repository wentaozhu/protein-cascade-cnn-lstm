
# coding: utf-8

# In[1]:

import numpy as np
cb6133 = np.load('cullpdb+profile_6133.npy')
cb6133 = np.reshape(cb6133, (6133, 700, 57))
dataindex = range(21)
labelindex = range(22,30)
maskindex = [30]
traindata = cb6133[:5600,:,dataindex]
trainlabel = cb6133[:5600,:,labelindex]
valdata = cb6133[5600:5877,:,dataindex]
vallabel = cb6133[5600:5877,:,labelindex]

traindata = np.concatenate((traindata, valdata), axis=0)
trainlabel = np.concatenate((trainlabel, vallabel), axis=0)
#trainlabel = trainlabel[:,:692,:]
testdata = cb6133[5877:,:,dataindex]
testlabel = cb6133[5877:,:,labelindex]

print(testlabel[2,691,:],testlabel[2,0,:],traindata.shape,valdata.shape,testdata.shape,trainlabel.shape,vallabel.shape,testlabel.shape)

import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Embedding, LSTM, Dense, merge, Convolution2D, Lambda, GRU, TimeDistributedDense, \
Reshape, Permute, Convolution1D, Masking, AveragePooling1D, MaxPooling1D
from keras.optimizers import Adam
from keras.regularizers import WeightRegularizer,l2
from keras.callbacks import ModelCheckpoint



# In[10]:

lr = 1e-3
nep = 100
seq = Sequential()
seq.add(Convolution1D(2, 5, input_shape=traindata.shape[1:],activation='tanh', border_mode='same', W_regularizer=l2(0.001)))# 80
seq.add(AveragePooling1D(pool_length=5, stride=1, border_mode='same'))
seq.add(Convolution1D(2, 5, activation='tanh', border_mode='same', W_regularizer=l2(0.001))) # 80
seq.add(AveragePooling1D(pool_length=5, stride=1, border_mode='same'))
seq.add(Convolution1D(2, 4, activation='tanh', border_mode='same', W_regularizer=l2(0.001))) # 80
seq.add(TimeDistributedDense(8,activation='softmax', name='output', W_regularizer=l2(0.001)))
adam = Adam(lr=lr)
seq.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
seq.summary()
#best_model_file = './onehotcnn'+str(lr)+str(nep)+'.h5' 
#best_model = ModelCheckpoint(best_model_file, monitor='val_output_acc', verbose=1, save_best_only=True) 
# and trained it via:
seq.fit(traindata,trainlabel,nb_epoch=nep, batch_size=2, 
          validation_data=(testdata,testlabel))