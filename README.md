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

cb6133idp538seq3.py is to use cb6133 + idp538 to do Q3 task. Other settings are the same as seq8

cb6133idp538seqpssm3.py is to use cb6133+idp538 to do Q3 task. Other settings are the same as seqpssm8

cb6133seq3.py is to use cb6133 to do Q3 task. Other settings are the same as seq8 and cb6133idp538seq3.py

cb6133seqpssm3.py is to use cb6133 to do Q3 task. Other seetings are the same as seqpssm8 and cb6133idp538seq3.py

cb6133seqpssmmultitask3.py is to use cb6133 to do Q3 multi-class task. Other settings are the same as seqpssmmultitask8.

test*.py is for getting the predicted labels and calculate the precision, recall.

Pretrained weights for the three models can be downloaded from https://drive.google.com/open?id=0B5Hl9mO74DHvT2VIWGVkMG5XU2s

Extra dataset can be downloaded from https://drive.google.com/open?id=0B5Hl9mO74DHvT2VIWGVkMG5XU2s
