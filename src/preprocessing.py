# Preprocessing module
## Loads COCO images and annotations and perform necessary preprocessing.
## Since the individual datasets are <20GB (smaller after filtering & resizing),
## we opt to simply store everything in an ndarray. If more data is present
## we should probably use the `tf.data` API.

import numpy as np
from warnings import warn
from os import path
from skimage.io import imread
from skimage.transform import resize
from pycocotools.coco import COCO
from config import NUM_KEYPOINTS, image_shape, sigma
from numba import jit

@jit
def update_confidence_map_internal(Y, x_kp, y_kp, sigma):
    # Calculate using a Gaussian kernel and take max
    # TODO: vectorize this part
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = max(Y[i,j], np.exp(-((i - x_kp)**2 + (j - y_kp)**2) / sigma**2))

def update_confidence_map_resize(img_shape, Y, annos, sigma, image_path):
    """
    Updates the w' x h' x NUM_KEYPOINTS confidence maps Y using one person's
    keypoints data.
    """
    Y_tmp = np.zeros((img_shape[0], img_shape[1], NUM_KEYPOINTS))
   
   
    for anno in annos:
        keypoints = anno["keypoints"]
        if len(keypoints) != 3 * NUM_KEYPOINTS:
            warn(f"Keypoints data for {image_path} is corrupted.")
            continue
        for k in range(NUM_KEYPOINTS):
            y_kp, x_kp, visibility = keypoints[3*k], keypoints[3*k+1], keypoints[3*k+2]
            if visibility == 2: # labeled and visible
                update_confidence_map_internal(Y_tmp[...,k], x_kp, y_kp, sigma)
    Y[:,:,:] = resize(Y_tmp, Y.shape, mode='reflect', anti_aliasing=True)   
    
def update_confidence_map(Y, keypoints, scale, sigma, image_path):
    """
    Updates the w' x h' x NUM_KEYPOINTS confidence maps Y using one person's
    keypoints data.
    """
    # Safety check
    if len(keypoints) != 3 * NUM_KEYPOINTS:
        warn(f"Keypoints data for {image_path} is corrupted.")
        return
    for k in range(NUM_KEYPOINTS):
        y_kp, x_kp, visibility = keypoints[3*k], keypoints[3*k+1], keypoints[3*k+2] # COCO keypoints are transposed
        if visibility == 2: # labeled and visible
            # Scale the coordinates
            x_kp /= scale[0]
            y_kp /= scale[1]
            update_confidence_map_internal(Y[...,k], x_kp, y_kp, sigma)

def load_data(data_dir, data_type, image_shape=image_shape, sigma=sigma, num_input=None, verbose=False, 
             image_ids = None, use_resize = True):
    """
    Load raw data from disk and preprocess into feature and label tensors.

    - data_dir: path to COCO dataset

    - data_type: 'train2014', 'val2014', 'test2014' or 'test2015'

    - image_shape: standardized shape to resize raw images into. Default: (224,224).

    - sigma: width of the Gaussian kernel used in the construction of the ground truth
      confidence map. Default: 1.0.

    - num_input: how many inputs to use, or all if set to None. Default: None.
    """

    # Load metadata
    image_dir = path.join(data_dir, "images", data_type)
    anno_path = path.join(data_dir, "annotations", f"person_keypoints_{data_type}.json")
    coco = COCO(anno_path)
    if image_ids == None:
        image_ids = coco.getImgIds(
            catIds=coco.getCatIds(catNms=['person'])
        ) # only load person images
        image_ids = np.random.permutation(image_ids)
        if num_input != None:
            image_ids = image_ids[:num_input]
    #print(image_ids)
    images = coco.loadImgs(image_ids)

    # Allocate feature and label tensor
    ## For the confidence map, here we hard-code its shape as 1/8 of the input shape
    ## (as a result of three pooling layers in Vgg19).
    if image_shape[0] % 8 != 0 or image_shape[1] % 8 != 0:
        warn("Input shape not divisible by 8.")
    confmap_shape = (image_shape[0]//8, image_shape[1]//8)
    X = np.ndarray((len(images), *image_shape, 3))
    Y = np.zeros((len(images), *confmap_shape, NUM_KEYPOINTS)) # one confidence map for each keypoint

    # Build the tensors
    for i, img_data in enumerate(images):
        # Build X (feature; scaled and resized input image)
        img_path = path.join(image_dir, img_data['file_name'])
        verbose and print("Processing ", img_path)
        img = imread(img_path).astype(float) # scale pixel value to [0,255]
        if len(img.shape) == 2:
            # Handle black and white images
            img = np.tile(np.reshape(img, (*img.shape, 1)), 3)
        X[i,:,:,:] = resize(img, image_shape, mode='reflect', anti_aliasing=True)

        # Compute the scale between the raw image and the confidence maps.
        # Used to convert keypoint coordinates in the raw image to coordinates
        # in the confidence maps.
        scale = (img.shape[0] / confmap_shape[0], img.shape[1] / confmap_shape[1])
        annos = coco.loadAnns(coco.getAnnIds(imgIds=img_data['id']))
        if use_resize:
            # Build Y (label; ground truth confidence maps)
            update_confidence_map_resize(img.shape, Y[i,:,:,:], annos, sigma, img_path)
        else:
            for anno in annos: # each annotation corresponds to a different person
                update_confidence_map(Y[i,:,:,:], anno['keypoints'], scale, sigma, img_path)
    return X, Y
