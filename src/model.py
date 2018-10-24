import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from config import NUM_KEYPOINTS, image_shape, learning_rate, momentum

# Build and compile model (up to stage1/branch1)
model = Sequential([
    # First 10 layers of vgg
    Conv2D(64, (3,3), padding='same', activation=tf.nn.relu, input_shape=(*image_shape, 3)),
    Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    MaxPooling2D(padding='same'),
    Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    MaxPooling2D(padding='same'),
    Conv2D(256, (3,3), padding='same', activation=tf.nn.relu),
    Conv2D(256, (3,3), padding='same', activation=tf.nn.relu),
    Conv2D(256, (3,3), padding='same', activation=tf.nn.relu),
    Conv2D(256, (3,3), padding='same', activation=tf.nn.relu),
    MaxPooling2D(padding='same'),
    Conv2D(512, (3,3), padding='same', activation=tf.nn.relu),
    Conv2D(512, (3,3), padding='same', activation=tf.nn.relu),
    # Reduce number of feature maps
    Conv2D(256, (3,3), padding='same', activation=tf.nn.relu),
    Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    
    # First branch (confidence map) of stage 1
    Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    Conv2D(512, (1,1), padding='valid', activation=tf.nn.relu),
    Conv2D(NUM_KEYPOINTS, (1,1), padding='valid')
])

## Optimizer: stochastic gradient descent with momentum
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # from keras documentation
sgd = SGD(lr=learning_rate, momentum=momentum)
model.compile(loss='mse', optimizer=sgd, metrics=['mae', 'acc'])
