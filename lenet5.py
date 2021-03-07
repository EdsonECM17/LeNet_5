# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:49:31 2021

@author: ecastaneda

LENET-5

"""

import tensorflow as tf
from tensorflow import keras


def build_lenet(activation):
    lenet_5_model = keras.models.Sequential()
    # C1 (Convolutional Layer 1)
    lenet_5_model.add(keras.layers.Conv2D(filters=6, kernel_size=5,
                                          strides=1, activation=activation,
                                          input_shape=(32, 32, 1),
                                          padding='valid',
                                          name='C1'))
    # S2 (Sub-sampling layer)
    lenet_5_model.add(tf.keras.layers.AveragePooling2D(pool_size=2,
                                                       strides=2,
                                                       name='S2'))
    # C3
    lenet_5_model.add(keras.layers.Conv2D(filters=16, kernel_size=5,
                                          strides=1, activation=activation,
                                          padding='valid',
                                          name='C3'))
    # S4
    lenet_5_model.add(keras.layers.AveragePooling2D(pool_size=2,
                                                    strides=2,
                                                    name='S4'))
    # C5 (Dense layer)
    lenet_5_model.add(keras.layers.Conv2D(filters=120, kernel_size=5,
                                          strides=1, activation=activation,
                                          padding='valid',
                                          name='C5'))
    lenet_5_model.add(keras.layers.Flatten(name='C5_F'))
    # F6 (Dense layer)
    lenet_5_model.add(keras.layers.Dense(84, activation=activation,
                                         name='F6'))

    # Output
    lenet_5_model.add(keras.layers.Dense(10, activation='softmax',
                                         name='output'))

    # Summary of model
    lenet_5_model.summary()

    lenet_5_model.compile(optimizer=keras.optimizers.Adam(),
                          loss=keras.losses.sparse_categorical_crossentropy,
                          metrics=['accuracy'])

    return lenet_5_model
