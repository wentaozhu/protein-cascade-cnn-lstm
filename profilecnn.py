
# coding: utf-8

# In[1]:

import numpy as np
cb6133 = np.load('cullpdb+profile_6133.npy')
cb6133 = np.reshape(cb6133, (6133, 700, 57))
dataindex = range(35,57)
labelindex = range(22,30)
maskindex = [30]
traindata = cb6133[:5600,:,dataindex]
trainlabel = cb6133[:5600,:,labelindex]
valdata = cb6133[5600:5877,:,dataindex]
vallabel = cb6133[5600:5877,:,labelindex]

traindata = np.concatenate((traindata, valdata), axis=0)
trainlabel = np.concatenate((trainlabel, vallabel), axis=0)

testdata = cb6133[5877:,:,dataindex]
testlabel = cb6133[5877:,:,labelindex]

print(testlabel[2,699,:],testlabel[2,0,:],traindata.shape,valdata.shape,testdata.shape,trainlabel.shape,vallabel.shape,testlabel.shape)

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

lr = 1e-3
nep = 150
l2value = 0
hsize = 600 #900
input = Input(shape=(700,22), name='input')
conv1 = Convolution1D(hsize, 5, activation='tanh', border_mode='same')(input)
pool1 = AveragePooling1D(pool_length=5, stride=1, border_mode='same')(conv1)
conv2 = Convolution1D(hsize, 5, activation='tanh', border_mode='same', W_regularizer=l2(l2value))(pool1)
pool2 = AveragePooling1D(pool_length=5, stride=1, border_mode='same')(conv2)
conv3 = Convolution1D(hsize, 4, activation='tanh', border_mode='same', W_regularizer=l2(l2value))(pool2)
output = TimeDistributedDense(8,activation='softmax', W_regularizer=l2(l2value), name='output')(conv3)
model = Model(input=input, output=output)
adam = Adam(lr=lr)
model.compile(optimizer=adam,
              loss={'output': 'categorical_crossentropy'},
              metrics=['weighted_accuracy'])
model.summary()
best_model_file = './profilecnn'+str(lr)+str(nep)+str(l2value)+str(hsize)+'.h5' 
best_model = ModelCheckpoint(best_model_file, monitor='val_weighted_accuracy', verbose=2, save_best_only=True) 
# and trained it via:
model.fit({'input': traindata},
          {'output': trainlabel},nb_epoch=nep, batch_size=128, 
          validation_data=({'input': testdata},
                           {'output': testlabel}), callbacks=[best_model], verbose=2)


# In[ ]:



