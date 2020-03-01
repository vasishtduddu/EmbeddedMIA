# From https://www.kaggle.com/arpandhatt/dense-sparse-dense-convolutional-neural-network

import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from skimage.io import imshow
from keras import backend as K
from keras.constraints import Constraint
from keras.datasets import fashion_mnist


img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


class Sparse(Constraint):
    '''
    We will use one variable: Mask
    After we train our model dense model,
    we will save the weights and analyze them.
    We will create a mask where 1 means the
    number is far away enough from 0 and 0
    if it is to close to 0. We will multiply
    the weights by 0(making them 0) if they
    are supposed to be masked.
    '''

    def __init__(self, mask):
        self.mask = K.cast_to_floatx(mask)

    def __call__(self,x):
        return self.mask * x

    def get_config(self):
        return {'mask': self.mask}


model = Sequential([
    Conv2D(32,3,input_shape=(28,28,1)),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64,3),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(250),
    Activation('relu'),
    Dropout(0.4),
    Dense(10),
    Activation('softmax')
])

adam = Adam()
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

# Dense train
model.fit(X_train_imgs[:41000], Y_train_oh[:41000],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train_imgs[41001:], Y_train_oh[41001:]))

# Sparsity Masks
def create_sparsity_masks(model,sparsity):
    weights_list = model.get_weights()
    masks = []
    for weights in weights_list:
        #We can ignore biases
        if len(weights.shape) > 1:
            weights_abs = np.abs(weights)
            masks.append((weights_abs>np.percentile(weights_abs,sparsity))*1.)
    return masks

masks = create_sparsity_masks(model,30)#Closest 30% to 0


# Sparse Model
sparse_model = Sequential([
    Conv2D(32,3,input_shape=(28,28,1), kernel_constraint=Sparse(masks[0])),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64,3, kernel_constraint=Sparse(masks[1])),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(250, kernel_constraint=Sparse(masks[2])),
    Activation('relu'),
    Dropout(0.4),
    Dense(10, kernel_constraint=Sparse(masks[3])),
    Activation('softmax')
])

adam = Adam()
sparse_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
sparse_model.summary()
#Get weights from densely trained model
sparse_model.set_weights(model.get_weights())


# Train Sparse Model
sparse_model.fit(X_train_imgs[:41000], Y_train_oh[:41000],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train_imgs[41001:], Y_train_oh[41001:]))


# Dense Training

redense_model = Sequential([
    Conv2D(32,3,input_shape=(28,28,1)),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64,3),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(250),
    Activation('relu'),
    Dropout(0.4),
    Dense(10),
    Activation('softmax')
])

adam = Adam(lr=0.0001)#Default Adam lr is 0.001 so I set it to 0.0001
redense_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
redense_model.summary()
#Get weights from sparsely trained model
redense_model.set_weights(sparse_model.get_weights())

redense_model.fit(X_train_imgs[:41000], Y_train_oh[:41000],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train_imgs[41001:], Y_train_oh[41001:]))
