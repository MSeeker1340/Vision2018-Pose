import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from config import NUM_KEYPOINTS, image_shape, learning_rate, momentum

# Full model (up to stage1/branch1, including the truncated VGG19)
full_model = Sequential([
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

# Model for stage1/branch1 only. Relies on a pre-trained VGG19 prestage
# and can only work for input_shape == (224, 224)
stage1 = Sequential([
    # Reduce number of feature maps
    Conv2D(256, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 512)),
    Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    
    # First branch (confidence map) of stage 1
    Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    Conv2D(512, (1,1), padding='valid', activation=tf.nn.relu),
    Conv2D(NUM_KEYPOINTS, (1,1), padding='valid')
])
    

# Optimizer
# optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # from keras documentation
# optimizer = SGD(lr=learning_rate, momentum=momentum) # same as openpose-plus
optimizer = Adam(lr=learning_rate)
full_model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
stage1.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
