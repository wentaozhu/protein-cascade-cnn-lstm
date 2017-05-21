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
cb513 = np.load('./dataset/idp/pssm/cb513_3state_f46.npy')
cb6133 = np.load('./dataset/idp/pssm/cb6133_3state_f46.npy')
idp546 = np.load('./dataset/idp/pssm/idp538_3state_f46.npy')

# In[2]:

print(cb513.shape)
print(cb6133.shape)
print(idp546.shape)
# print(len(idp546[0]), len(idp546))
# idp546arr = np.zeros((len(idp546), len(idp546[0])))
# for i in xrange(idp546.shape[0]):
#   idp546arr[i,:] = np.asarray(idp546[i])
# idp546 = np.array(idp546arr)
# print(idp546.shape)
# In[7]:

cb513reshape = np.reshape(cb513, (cb513.shape[0], 700, 46))
cb6133reshape = np.reshape(cb6133, (cb6133.shape[0], 700, 46))
idp546reshape = np.reshape(idp546, (idp546.shape[0], 700, 46))
print(cb513reshape.shape, cb6133reshape.shape, idp546reshape.shape)

# In[8]:

print(cb513reshape[0,0,30], cb6133reshape[0,0,30], cb513reshape[0,-1,30], cb6133reshape[0,-2,30])


# In[12]:

dataindex = range(42) # only use sequence
labelindex = range(42,45) # q 3
maskindex = [45] # noseq
solvindex = [46, 47] # solvent accessibility
traindata = cb6133reshape[:,:,dataindex]
trainlabel = cb6133reshape[:,:,labelindex]
idp546data = idp546reshape[:,:,dataindex]
idp546label = idp546reshape[:,:,labelindex]
traindataext = np.zeros((traindata.shape[0]+idp546data.shape[0], traindata.shape[1], traindata.shape[2]))
trainlabelext = np.zeros((trainlabel.shape[0]+idp546label.shape[0], trainlabel.shape[1], trainlabel.shape[2]))
traindataext[:traindata.shape[0],:,:] = np.array(traindata)
traindataext[traindata.shape[0]:,:,:] = np.array(idp546data)
trainlabelext[:traindata.shape[0],:,:] = np.array(trainlabel)
trainlabelext[traindata.shape[0]:,:,:] = np.array(idp546label)
traindata = np.array(traindataext)
trainlabel = np.array(trainlabelext)

trainmask = cb6133reshape[:,:,maskindex]* -1 + 1 # 0 for no seq, 1 for seq
idp546mask = idp546reshape[:,:,maskindex]* -1 + 1
trainmaskext = np.zeros((trainmask.shape[0]+idp546mask.shape[0], trainmask.shape[1], trainmask.shape[2]))
trainmaskext[:trainmask.shape[0],:,:] = np.array(trainmask)
trainmaskext[trainmask.shape[0]:,:,:] = np.array(idp546mask)
trainmask = np.array(trainmaskext)
valdata = cb513reshape[:,:,dataindex]
vallabel = cb513reshape[:,:,labelindex]

valmask = cb513reshape[:,:,maskindex] * -1 + 1

traindataaux = traindata[:,:,21:] #22
traindata = traindata[:,:,:21]

valdataaux = valdata[:,:,21:] # 2
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
auxiliary_input = Input(shape=(700,21), name='aux_input')  #24
#auxiliary_input = Masking(mask_value=0)(auxiliary_input)

x = Concatenate(axis=-1)([x,auxiliary_input])
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
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
adam = Adam(lr=0.0003)
model.compile(optimizer=adam,
              loss={'main_output': 'categorical_crossentropy'},
              loss_weights={'main_output': 1.},
              metrics=['weighted_accuracy'])
model.summary()
best_model_file = './bestacccb6133idp538seqpssm33e-4.h5' 
print(best_model_file)
plot_model(model, to_file=best_model_file[:-2]+'png', show_shapes=True)
best_model = ModelCheckpoint(best_model_file, monitor='val_weighted_accuracy', verbose=1, save_best_only = True) 
# and trained it via:
print(traindataint.shape, traindataaux.shape, trainlabel.shape, valdataint.shape, valdataaux.shape, vallabel.shape)
model.fit({'main_input': traindataint, 'aux_input': traindataaux},
          {'main_output': trainlabel},epochs=150, batch_size=96, 
          validation_data=({'main_input': valdataint, 'aux_input': valdataaux},
                           {'main_output': vallabel}), callbacks=[best_model], verbose=2)

