"""
ResNet model for regression of Keras.
"Liu, C.; Zhang, H.R.; Cheng, Z. Emulation of an atmospheric gas-phase chemistry solver through deep learning. Atmospheric Pollution Research, 2021"
"""

import numpy as np
import keras
import tensorflow as tf
import os
from keras.models import Model
from keras.layers import Input, Dense, add, ReLU, Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def dataGenerator(x_train, y_train, batch_size):
    while True:   
        for i in range(np.ceil(1.0*x_train.shape[0]/batch_size).astype(int)):
            x_train_std = (x_train[i*batch_size:(i+1)*batch_size] - x_train_min) / dx
            y_train_std = (y_train[i*batch_size:(i+1)*batch_size] - y_train_min) / dy
            yield x_train_std, y_train_std


os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
batch_size = 4096

x_train = np.load('X_all.npy', mmap_mode='r')
y_train = np.load('Y_all.npy', mmap_mode='r')

x_train_min = np.load('X_train_min.npy')
x_train_max = np.load('X_train_max.npy')
y_train_min = np.load('Y_train_min_d.npy')
y_train_max = np.load('Y_train_max_d.npy')

dx = x_train_max - x_train_min
dy = y_train_max - y_train_min

x_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy')
x_test_std = (x_test - x_train_min) / dx
y_test_std = (y_test - y_train_min) / dy 

    
print('begin')
input0 = Input(shape=(236,))

out1 = Dense(units=1024)(input0)
out1 = Activation('relu')(out1)
out2 = Dense(units=512)(out1)
out2 = Activation('relu')(out2)
out3 = Dense(units=236)(out2)
out3 = add([out3, input0])
out3 = Activation('relu')(out3)

out4 = Dense(units=1024)(out3)
out4 = Activation('relu')(out4)
out5 = Dense(units=512)(out4)
out5 = Activation('relu')(out5)
out6 = Dense(units=236)(out5)
out6 = add([out6, out3])
out6 = Activation('relu')(out6)

out7 = Dense(units=1024)(out6)
out7 = Activation('relu')(out7)
out8 = Dense(units=512)(out7)
out8 = Activation('relu')(out8)
out9 = Dense(units=236)(out8)
out9 = add([out9, out6])
out9 = Activation('relu')(out9)

out10 = Dense(units=1024)(out9)
out10 = Activation('relu')(out10)
out11 = Dense(units=512)(out10)
out11 = Activation('relu')(out11)
out12 = Dense(units=236)(out11)
out12 = add([out12, out9])
out12 = Activation('relu')(out12)

out13 = Dense(units=1024)(out12)
out13 = Activation('relu')(out13)
out14 = Dense(units=512)(out13)
out14 = Activation('relu')(out14)
out15 = Dense(units=236)(out14)
out15 = add([out15, out12])
out15 = Activation('relu')(out15)

out16 = Dense(units=1024)(out15)
out16 = Activation('relu')(out16)
out17 = Dense(units=512)(out16)
out17 = Activation('relu')(out17)
out18 = Dense(units=236)(out17)
out18 = add([out18, out15])
out18 = Activation('relu')(out18)

out19 = Dense(units=1024)(out18)
out19 = Activation('relu')(out19)
out20 = Dense(units=512)(out19)
out20 = Activation('relu')(out20)
out21 = Dense(units=236)(out20)
out21 = add([out21, out18])
out21 = Activation('relu')(out21)

out22 = Dense(units=1024)(out21)
out22 = Activation('relu')(out22)
out23 = Dense(units=512)(out22)
out23 = Activation('relu')(out23)
out24 = Dense(units=236)(out23)
out24 = add([out24, out21])
out24 = Activation('relu')(out24)

out25 = Dense(units=1024)(out24)
out25 = Activation('relu')(out25)
out26 = Dense(units=512)(out25)
out26 = Activation('relu')(out26)
out27 = Dense(units=236)(out26)
out27 = add([out27, out24])
out27 = Activation('relu')(out27)

out = Dense(units=194)(out27)
model = Model(inputs=input0, outputs=out)

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam())
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
filepath = 'DLemulator.h5'
checkpoint= ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit_generator(dataGenerator(x_train, y_train, batch_size), 
                    steps_per_epoch=int(np.ceil(1.0*x_train.shape[0]/batch_size)),
                    epochs=200, 
                    verbose=2, 
                    validation_data=(x_test_std, y_test_std), 
                    callbacks=[reduce_lr,checkpoint])
