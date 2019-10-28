'''Trains a simple binarize CNN on the FMNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 98.98% test accuracy after 20 epochs using tensorflow backend
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf
#np.random.seed(1337)  # for reproducibility

import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, AveragePooling2D, Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import top_k_categorical_accuracy
#from binary_ops import binary_tanh as binary_tanh_op
#from binary_layers import BinaryDense, BinaryConv2D

from tensorflow.keras.utils import plot_model
#from keras.utils import model_to_dot

#from IPython.display import SVG
#import graphviz
#import pydot_ng as pydot

##################################################################################################################################
# Def new functions
def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
##################################################################################################################################
H = 1.
kernel_lr_multiplier = 'Glorot'

# nn
batch_size = 50
epochs = 5
total_epochs = 15
channels = 1
img_rows = 28
img_cols = 28
filters = 32
kernel_size = (3, 3)
pool_size = (2, 2)
hidden_units = 128
classes = 10
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
p1 = 0.25
p2 = 0.5

##################################################################################################################################
# the data, shuffled and split between train_f and test_f sets
(X_train_f, y_train_f), (X_test_f, y_test_f) = fashion_mnist.load_data()

X_train_f = X_train_f.reshape(60000, 1, 28, 28)
X_test_f = X_test_f.reshape(10000, 1, 28, 28)
X_train_f = X_train_f.astype('float32')
X_test_f = X_test_f.astype('float32')
X_train_f /= 255
X_test_f /= 255
print(X_train_f.shape[0], 'train_f samples')
print(X_test_f.shape[0], 'test_f samples')

# convert class vectors to binary class matrices
#Y_train_f = np_utils.to_categorical(y_train_f, classes) * 2 - 1 # -1 or 1 for hinge loss
#Y_test_f = np_utils.to_categorical(y_test_f, classes) * 2 - 1

Y_train_f = to_categorical(y_train_f, classes)
Y_test_f  = to_categorical(y_test_f, classes)

##################################################################################################################################
input   = Input(shape=(1,28,28))

# conv1
conv1 = Conv2D(32, kernel_size, strides=(1, 1), padding='valid', data_format='channels_first',
        dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, name='conv1')(input)
pool21= MaxPooling2D(pool_size=pool_size, name='pool21', data_format='channels_first')(conv1)
bn1   = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1')(pool21)
act1  = Activation(relu6, name='act1')(bn1)

# conv2
conv2 = Conv2D(64, kernel_size, strides=(1, 1), padding='valid', data_format='channels_first',
        dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, name='conv2')(act1)
pool22= MaxPooling2D(pool_size=pool_size, name='pool22', data_format='channels_first')(conv2)
bn2   = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2')(pool22)
act2  = Activation(relu6, name='act2')(bn2)



# conv4
conv3 = Conv2D(64, kernel_size, strides=(1, 1), padding='valid', data_format='channels_first',
        dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, name='conv4')(act2)
#bpool1= MaxPooling2D(pool_size=pool_size, name='bpool1', data_format='channels_first')(conv4)
bn3   = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn4')(conv3)
act3  = Activation(relu6, name='act4')(bn3)

# conv5
conv4 = Conv2D(128, kernel_size, strides=(1, 1), padding='valid', data_format='channels_first',
        dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, name='conv5')(act3)
#bpool2= MaxPooling2D(pool_size=pool_size, name='bpool2', data_format='channels_first')(conv5)
bn4   = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn5')(conv4)
act4  = Activation(relu6, name='act5')(bn4)


#flatten_2
flat2= Flatten()(act4)
#Dense_21
dns21 = Dense(128, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None, name='dns21')(flat2)
bn_d3 = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn_d3')(dns21)
act_d3= Activation(relu6, name='act_d3')(bn_d3)

#Dense_22
dns22 = Dense(classes, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None, name='dns22')(act_d3)
bn_d4 = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn_d4')(dns22)
b_dro = Dropout(0.01)(bn_d4)
act_d4= Activation('softmax', name='act_d4')(b_dro)

##################################################################################################################################
# model generation
merged2 = Model(inputs=[input],outputs=[act_d4])

##################################################################################################################################
# model compilation
#opt = Adam(lr=lr_start)
opt = Adam()
earlystop = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
merged2.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy', top_3_accuracy])

#merged2.compile(loss= keras.losses.categorical_crossentropy, optimizer= keras.optimizers.Adadelta(), metrics=['accuracy'])

#merged2.compile(loss='squared_hinge', optimizer=opt, metrics=['accuracy'])

##################################################################################################################################
# model details
merged2.summary()

##################################################################################################################################
# model visualization

##################################################################################################################################
# model fit
##### Begin Batch Train #####

#lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
fmnist        = 'training_fmnist' + '_baseline_' + '.log'
csv_logger_f  = CSVLogger(fmnist)
reduce_lr_f   = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                          patience=2, min_lr=0.0001)

print ("#################### Training Fashion MNIST Dataset ####################")
history1 = merged2.fit(X_train_f, Y_train_f,
                batch_size=batch_size, epochs=epochs,
                verbose=1, validation_data=(X_test_f, Y_test_f),
                callbacks=[reduce_lr_f])
##### End Batch Train #####

##################################################################################################################################
# Test Scoring
score2 = merged2.evaluate(X_test_f, Y_test_f, verbose=0)

##################################################################################################################################
# model details
merged2.summary()

print('Test score baseline fashion mnist:', score2[0])
print('Test accuracy baseline fashion mnist:', score2[1])

##################################################################################################################################
## model saving
merged2.save('fmnist_baseline.h5')

##################################################################################################################################
