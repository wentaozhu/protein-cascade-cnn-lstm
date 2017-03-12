
# coding: utf-8

# In[5]:

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
#trainlabel = trainlabel[:,:175,:]

testdata = cb6133[5877:,:,dataindex]
testlabel = cb6133[5877:,:,labelindex]

print(testlabel[2,174,:],testlabel[2,0,:],traindata.shape,valdata.shape,testdata.shape,trainlabel.shape,vallabel.shape,testlabel.shape)

import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Embedding, LSTM, Dense, merge, Convolution2D, Lambda, GRU, TimeDistributedDense, Reshape, Permute, Convolution1D, Masking, AveragePooling1D
from keras.optimizers import Adam
from keras.regularizers import WeightRegularizer,l2
from keras.callbacks import ModelCheckpoint



# In[9]:

lr = 1e-3
nep = 200 #150
l2value = 0
hsize = 600
input = Input(shape=(700,21), name='input')
conv1 = Convolution1D(100, 5, activation='tanh', border_mode='same', W_regularizer=l2(l2value))(input)
pool1 = AveragePooling1D(pool_length=5, stride=1, border_mode='same')(conv1)
conv2 = Convolution1D(120, 5, activation='tanh', border_mode='same', W_regularizer=l2(l2value))(pool1)
pool2 = AveragePooling1D(pool_length=5, stride=1, border_mode='same')(conv2)
conv3 = Convolution1D(160, 4, activation='tanh', border_mode='same', W_regularizer=l2(l2value))(pool2)
output = TimeDistributedDense(8,activation='softmax', name='output', W_regularizer=l2(l2value))(conv3)
model = Model(input=input, output=output)
adam = Adam(lr=lr)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['weighted_accuracy'])
model.summary()
best_model_file = './onehotcnn'+str(lr)+str(nep)+str(l2value)+'.h5'#+str(hsize)+'tanh.h5' 
best_model = ModelCheckpoint(best_model_file, monitor='val_weighted_accuracy', verbose=2, save_best_only=True) 
# and trained it via:
model.fit(traindata,trainlabel,nb_epoch=nep, batch_size=128, 
          validation_data=(testdata,testlabel), callbacks=[best_model], verbose=2)


# In[ ]: