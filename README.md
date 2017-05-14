# protein-cascade-cnn-lstm
Implementation of IJCAI16 cascade cnn and LSTM for protein secondary structure prediction
https://arxiv.org/abs/1604.07176
Protein Secondary Structure Prediction Using Cascaded Convolutional and Recurrent Neural Networks

The report is Noble Kennamer and Wentao Zhu, "Protein Secondary Structure Prediction - A Comprehensive Comparison", UCI, 2017.


If you have any questions, please contact with me wentaozhu1991@gmail.com.

The evaluation metric should be weighted accuracy (because there are many blank state in the data), which is in the metrics.py. You can copy the function and paste it into the keras metric.py. Then compile keras, install. 

# keras 2.0.3 and theano backend -- The latest version

First, download the files from http://www.princeton.edu/~jzthree/datasets/ICML2014/

seq 8 is to use 21 amino acid to predict 8 classes.

seqpssm8 is to use 21 amino acid and 21 sequence profile to predict 8 classes

seqpssmmultitask8 is to use 21 amino acid and 21 sequence profile to predict 8 classes and solvent property.
