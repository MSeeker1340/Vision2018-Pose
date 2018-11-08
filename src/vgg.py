import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

# Get pretrained VGG19 model
vgg19 = VGG19(weights='imagenet')

# We only want the first ten layers
truncated_vgg19 = Model(
    inputs=vgg19.input,
    outputs=vgg19.get_layer('block4_conv2').output
)

"""
    prestage(X) -> Z
    
Pass the #examples x 224 x 224 x 3 input tensor X
through a pretrained VGG19 model and get result of
the 10th layer as Z (#exmamples x 28 x 28 x 512).

Note that X is modified in place.
"""
def prestage(X):
    preprocess_input(X)
    return truncated_vgg19.predict(X)
