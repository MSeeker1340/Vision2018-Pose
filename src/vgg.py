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
    normalize_per_channel(Z,[factors]) -> factors
    
Normalize Z per channel (last dimension) by shifting
every channel by its mean and then divide by its standard
deviation. Returns the 2xN normalization factors where

  * factors[0,:] are the original means for every channel;
  * factors[1,:] are the original stds for every channel.

If factors is given, then Z is normalized using the given
factors instead.

TODO: this is very slow and there's likely optimized method
in tensorflow that we can use instead. (keras.BatchNormalization?)
"""
def normalize_per_channel(Z, factors=None):
    num_channels = Z.shape[-1]
    if factors is None:
        # Compute means and stds
        factors = np.zeros((2,num_channels))
        for ch in range(num_channels):
            factors[0,ch] = np.mean(Z[...,ch])
            factors[1,ch] = np.std(Z[...,ch])
    for ch in range(num_channels):
        Z[...,ch] -= factors[0,ch]
        Z[...,ch] /= factors[1,ch]
    return factors

"""
    prestage(X) -> Z
    
Pass the #examples x 224 x 224 x 3 input tensor X
through a pretrained VGG19 model and get result of
the 10th layer as Z (#exmamples x 28 x 28 x 512) as
well as the normalization factors.

First vgg19.preprocess_input is called on X to shift
the RGB channels to be zero-centered (this is the
expected input for VGG19). Output is then fetched by
feeding X to the truncated network.

Note that X is modified in place.
"""
def prestage(X):
    preprocess_input(X)
    return truncated_vgg19.predict(X)
