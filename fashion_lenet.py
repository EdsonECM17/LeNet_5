# -*- coding: utf-8 -*-
"""
Created on Sun Feb 7 23:35:06 2021

@author: ecastaneda
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

from lenet5 import build_lenet
from cnn_utilities import plot_results

# Get input data
(train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)

# Padding
# MNIST is 28x28 and LeNet-5 original CNN has a 32x32 input
train_x = np.pad(train_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
test_x = np.pad(test_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

# Validation
val_x = train_x[:5000]
val_y = train_y[:5000]

# Defining activation function
# activation = 'tanh'
activation = 'relu'

# Get model
fashion_lenet_model = build_lenet(activation)

# Plot model
if not os.path.exists("fashion"):
    os.mkdir('fashion')
keras.utils.plot_model(fashion_lenet_model, to_file='fashion/model.png',
                       show_shapes=True)

# Fit model intro training an validation data
history = fashion_lenet_model.fit(train_x, train_y, batch_size=128, epochs=40,
                                  verbose=1, validation_data=(val_x, val_y))

# Plot training results
plot_results(history, 'fashion/')

# Evaluate model
loss, accuracy = fashion_lenet_model.evaluate(test_x, test_y)

print('Test Loss:', loss)
print('Test accuracy:', accuracy)
