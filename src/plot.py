import numpy as np
import matplotlib.pyplot as pyplot
from skimage.transform import resize

def plot_result(y, f):
    rshape = y.shape
    ishape = (224, 224, 3)
    result = np.zeros(ishape)
    # scale = (ishape[0]/rshape[0], ishape[1]/rshape[1])
    #result[:,:,0] = resize(y[:,:,f].transpose(), (224, 224), mode='reflect', anti_aliasing=True)
    result[:,:,0] = resize(y[:,:,f], (224, 224), mode='reflect', anti_aliasing=True)
    #pyplot.imshow(result)
    return result

def plot_on_img(x, y, f):
    sigma = 0.05
    result = plot_result(y, f)
    img = np.zeros(x.shape)
    img[:,:,:] = x
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if result[i, j, 0] >= 0.05:
                img[i, j, 0] = min(result[i, j, 0]*3, 1.0)
                img[i, j, 1] = img[i, j, 1]/2
                img[i, j, 2] = img[i, j, 2]/2
    pyplot.imshow(img)
    
# plot_on_img(X[10], Y[10], 0)
    

