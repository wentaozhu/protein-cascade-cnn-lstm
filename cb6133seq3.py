# coding: utf-8

# In[1]:

import numpy as np
import os
from keras import backend as K
# import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# config = tf.ConfigProto()
# # config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)
cb513 = np.load('./dataset/cb6133+cb513_3state_with_sasa/cb513_3state_f48_sasa.npy')
cb6133 = np.load('./dataset/cb6133+cb513_3state_with_sasa/cb6133_3state_f48_sasa.npy')

# In[2]:

print(cb513.shape)
print(cb6133.shape)
# In[7]:

cb513reshape = np.reshape(cb513, (cb513.shape[0], 700, 48))
cb6133reshape = np.reshape(cb6133, (cb6133.shape[0], 700, 48))

print(cb513reshape.shape, cb6133reshape.shape)

# In[8]:

print(cb513reshape[0,0,30], cb6133reshape[0,0,30], cb513reshape[0,-1,30], cb6133reshape[0,-2,30])

# In[12]:

dataindex = range(21) # only use sequence
labelindex = range(42,45) # q 3
maskindex = [45] # noseq
traindata = cb6133reshape[:,:,dataindex]
trainlabel = cb6133reshape[:,:,labelindex]

trainmask = cb6133reshape[:,:,maskindex]* -1 + 1 # 0 for no seq, 1 for seq

valdata = cb513reshape[:,:,dataindex]
vallabel = cb513reshape[:,:,labelindex]

valmask = cb513reshape[:,:,maskindex] * -1 + 1
#traindata = np.concatenate((traindata, valdata), axis=0)
# traindataaux = traindata[:,:,24:-1] #22
traindata = traindata[:,:,:21]
#trainlabel = np.concatenate((trainlabel, vallabel), axis=0)
#trainsolvlabel = np.concatenate((trainsolvlabel, valsolvlabel), axis=0)
#trainmask = np.concatenate((trainmask, valmask), axis=0)
# valdataaux = valdata[:,:,24:-1] # 22
valdata = valdata[:,:,:21]

# convert one hot to interger
traindata = traindata[:,:,:21]
traindataint = np.ones((traindata.shape[0], traindata.shape[1]))
for i in xrange(traindata.shape[0]):
    for j in xrange(traindata.shape[1]):
        if np.sum(traindata[i,j,:]) != 0:
            traindataint[i,j] = np.argmax(traindata[i,j,:])
valdata = valdata[:,:,:21]
valdataint = np.ones((valdata.shape[0], valdata.shape[1]))
for i in xrange(valdata.shape[0]):
    for j in xrange(valdata.shape[1]):
        if np.sum(valdata[i,j,:]) != 0:
            valdataint[i,j] = np.argmax(valdata[i,j,:])
print(valdataint.max(), valdataint.min(), traindataint.max(), traindataint.min())
print(vallabel[2,699,:],vallabel[2,0,:])
print(traindata.shape, traindataint.shape)

# In[ ]:

import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Embedding, LSTM, Dense, Lambda, GRU, TimeDistributed, Reshape, Permute, Conv1D, Masking #TimeDistributedDense
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2 #WeightRegularizer
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
main_input = Input(shape=(700,), dtype='int32', name='main_input')
#main_input = Masking(mask_value=23)(main_input)
x = Embedding(output_dim=50, input_dim=21, input_length=700)(main_input)
# auxiliary_input = Input(shape=(700,21), name='aux_input')  #24
#auxiliary_input = Masking(mask_value=0)(auxiliary_input)
# x = merge([x, auxiliary_input], mode='concat', concat_axis=-1)
#x = Reshape((1,700,74))(x)
a = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
#a = Permute((2,3,1))(a)
b = Conv1D(64, 7, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
#b = Permute((2,3,1))(b)
c = Conv1D(64, 11, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
#c = Permute((2,3,1))(c)
x = Concatenate(axis=-1)([a,b,c])
#x = Reshape((700,192))(x)
d = GRU(units=300, return_sequences=True, dropout=0.5)(x)
e = GRU(units=300, return_sequences=True, go_backwards=True, dropout=0.5)(x)
f = Concatenate(axis=-1)([d,e])
d = GRU(units=300, return_sequences=True, dropout=0.5)(f)
e = GRU(units=300, return_sequences=True, go_backwards=True, dropout=0.5)(f)
f = Concatenate(axis=-1)([d,e])
d = GRU(units=300, return_sequences=True, dropout=0.5)(f)
e = GRU(units=300, return_sequences=True, go_backwards=True, dropout=0.5)(f)
f = Concatenate(axis=-1)([d,e,x])
f = TimeDistributed(Dense(200,activation='relu', kernel_regularizer=l2(0.001)))(f)
f = TimeDistributed(Dense(200,activation='relu', kernel_regularizer=l2(0.001)))(f)
main_output = TimeDistributed(Dense(3,activation='softmax'), name='main_output')(f)
# auxiliary_output = TimeDistributedDense(4,activation='softmax', name='aux_output')(f)
model = Model(inputs=[main_input], outputs=[main_output])
adam = Adam(lr=0.0003)
model.compile(optimizer=adam,
              loss={'main_output': 'categorical_crossentropy'},
              metrics=['weighted_accuracy'])
model.summary()
best_model_file = './bestacccb6133seq33e-4.h5' 
print(best_model_file)
plot_model(model, to_file=best_model_file[:-2]+'png', show_shapes=True)
best_model = ModelCheckpoint(best_model_file, monitor='val_weighted_accuracy', verbose=1, save_best_only = True) 
# and trained it via:
print(traindataint.shape, trainlabel.shape, valdataint.shape, vallabel.shape)
model.fit({'main_input': traindataint},
          {'main_output': trainlabel},epochs=150, batch_size=96, 
          validation_data=({'main_input': valdataint},
                           {'main_output': vallabel}), callbacks=[best_model], verbose=2)