
# coding: utf-8

# In[3]:

import numpy as np
cb6133 = np.load('cullpdb+profile_6133.npy')


# In[4]:

cb6133 = np.reshape(cb6133, (6133, 700, 57))


# In[5]:

dataindex = range(22)+range(31,33)+range(35,57)
labelindex = range(22,30)
solvindex = range(33,35)
maskindex = [30]
traindata = cb6133[:5600,:,dataindex]
trainlabel = cb6133[:5600,:,labelindex]
trainsolvlabel = cb6133[:5600,:,solvindex]
trainsolvvalue = trainsolvlabel[:,:,0]*2 + trainsolvlabel[:,:,1]
trainsolvlabel = np.zeros((trainsolvvalue.shape[0], trainsolvvalue.shape[1], 4))
for i in xrange(trainsolvvalue.shape[0]):
    for j in xrange(trainsolvvalue.shape[1]):
        if np.sum(trainlabel[i,j,:]) != 0:
            trainsolvlabel[i,j,trainsolvvalue[i,j]] = 1
trainmask = cb6133[:5600,:,maskindex]* -1 + 1
valdata = cb6133[5600:5877,:,dataindex]
vallabel = cb6133[5600:5877,:,labelindex]
valsolvlabel = cb6133[5600:5877,:,solvindex]
valsolvvalue = valsolvlabel[:,:,0]*2 + valsolvlabel[:,:,1]
valsolvlabel = np.zeros((valsolvvalue.shape[0], valsolvvalue.shape[1], 4))
for i in xrange(valsolvvalue.shape[0]):
    for j in xrange(valsolvvalue.shape[1]):
        if np.sum(vallabel[i,j,:]) != 0:
            valsolvlabel[i,j,valsolvvalue[i,j]] = 1
valmask = cb6133[5600:5877,:,maskindex] * -1 + 1
#traindata = np.concatenate((traindata, valdata), axis=0)
traindataaux = traindata[:,:,24:-1] #22
traindata = traindata[:,:,:22]
#trainlabel = np.concatenate((trainlabel, vallabel), axis=0)
#trainsolvlabel = np.concatenate((trainsolvlabel, valsolvlabel), axis=0)
#trainmask = np.concatenate((trainmask, valmask), axis=0)
valdataaux = valdata[:,:,24:-1] # 22
valdata = valdata[:,:,:22]
testdata = cb6133[5877:,:,dataindex]
testlabel = cb6133[5877:,:,labelindex]
testsolvlabel = cb6133[5877:,:,solvindex]
testsolvvalue = testsolvlabel[:,:,0]*2 + testsolvlabel[:,:,1]
testsolvlabel = np.zeros((testsolvvalue.shape[0], testsolvvalue.shape[1], 4))
for i in xrange(testsolvvalue.shape[0]):
    for j in xrange(testsolvvalue.shape[1]):
        if np.sum(testlabel[i,j,:]) != 0:
            testsolvlabel[i,j,testsolvvalue[i,j]] = 1
testmask = cb6133[5877:,:,maskindex] * -1 + 1
testdataaux = testdata[:,:,24:-1] #22
testdata = testdata[:,:,:22] #22
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
traindataint = np.concatenate((traindataint, valdataint), axis=0)
traindataaux = np.concatenate((traindataaux, valdataaux), axis=0)
#traindataaux[:,:,-1] = 1-traindataaux[:,:,-1]
print(traindataaux[7,0,:],traindataaux[7,699,:])
trainlabel = np.concatenate((trainlabel, vallabel), axis=0)
trainsolvlabel = np.concatenate((trainsolvlabel, valsolvlabel), axis=0)
testdata = testdata[:,:,:21]
testdataint = np.ones((testdata.shape[0], testdata.shape[1]))
for i in xrange(testdata.shape[0]):
    for j in xrange(testdata.shape[1]):
        if np.sum(testdata[i,j,:]) != 0:
            testdataint[i,j] = np.argmax(testdata[i,j,:])
#testdataaux[:,:,-1] = 1-testdataaux[:,:,-1]
print(testdataint.max(), testdataint.min(), traindataint.max(), traindataint.min())
print(testlabel[2,699,:],testlabel[2,0,:])


# In[ ]:

import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Embedding, LSTM, Dense, merge, Convolution2D, Lambda, GRU, TimeDistributedDense, Reshape, Permute, Convolution1D, Masking
from keras.optimizers import Adam
from keras.regularizers import WeightRegularizer,l2
from keras.callbacks import ModelCheckpoint
traindata.shape


# In[ ]:

main_input = Input(shape=(700,), dtype='int32', name='main_input')
#main_input = Masking(mask_value=23)(main_input)
x = Embedding(output_dim=50, input_dim=21, input_length=700)(main_input)
auxiliary_input = Input(shape=(700,21), name='aux_input')  #24
#auxiliary_input = Masking(mask_value=0)(auxiliary_input)
x = merge([x, auxiliary_input], mode='concat', concat_axis=-1)
#x = Reshape((1,700,74))(x)
a = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(0.001))(x)
#a = Permute((2,3,1))(a)
b = Convolution1D(64, 7, activation='relu', border_mode='same', W_regularizer=l2(0.001))(x)
#b = Permute((2,3,1))(b)
c = Convolution1D(64, 11, activation='relu', border_mode='same', W_regularizer=l2(0.001))(x)
#c = Permute((2,3,1))(c)
x = merge([a,b,c], mode='concat', concat_axis=-1)
#x = Reshape((700,192))(x)
d = GRU(output_dim=300, return_sequences=True, activation='tanh', inner_activation='sigmoid', dropout_W=0.5)(x)
e = GRU(output_dim=300, return_sequences=True, activation='tanh', inner_activation='sigmoid', go_backwards=True, dropout_W=0.5)(x)
f = merge([d,e], mode='concat')
d = GRU(output_dim=300, return_sequences=True, activation='tanh', inner_activation='sigmoid', dropout_W=0.5)(f)
e = GRU(output_dim=300, return_sequences=True, activation='tanh', inner_activation='sigmoid', go_backwards=True, dropout_W=0.5)(f)
f = merge([d,e], mode='concat')
d = GRU(output_dim=300, return_sequences=True, activation='tanh', inner_activation='sigmoid', dropout_W=0.5)(f)
e = GRU(output_dim=300, return_sequences=True, activation='tanh', inner_activation='sigmoid', go_backwards=True, dropout_W=0.5)(f)
f = merge([d,e,x], mode='concat')
f = TimeDistributedDense(200,activation='relu', W_regularizer=l2(0.001))(f)
f = TimeDistributedDense(200,activation='relu', W_regularizer=l2(0.001))(f)
main_output = TimeDistributedDense(8,activation='softmax', name='main_output')(f)
auxiliary_output = TimeDistributedDense(4,activation='softmax', name='aux_output')(f)
model = Model(input=[main_input, auxiliary_input], output=[main_output, auxiliary_output])
adam = Adam(lr=0.0003)
model.compile(optimizer=adam,
              loss={'main_output': 'categorical_crossentropy', 'aux_output': 'categorical_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 1.},
              metrics=['weighted_accuracy'])
model.summary()
best_model_file = './ijcaibestacctwomasknowunoembedsoftmaxregular3e-4.h5' 
best_model = ModelCheckpoint(best_model_file, monitor='val_main_output_weighted_accuracy', verbose=1, save_best_only = True) 
# and trained it via:
print(traindataint.shape, traindataaux.shape, trainlabel.shape, trainsolvlabel.shape, testdataint.shape, testdataaux.shape,
     testlabel.shape, testsolvlabel.shape)
model.fit({'main_input': traindataint, 'aux_input': traindataaux},
          {'main_output': trainlabel, 'aux_output': trainsolvlabel},nb_epoch=100, batch_size=96, 
          validation_data=({'main_input': testdataint, 'aux_input': testdataaux},
                           {'main_output': testlabel, 'aux_output': testsolvlabel}), callbacks=[best_model], verbose=2)


# In[ ]:



