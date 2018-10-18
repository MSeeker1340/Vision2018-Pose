# Preprocessing module
## Loads COCO images and annotations and perform necessary preprocessing.
## Since the individual datasets are <20GB (smaller after filtering & resizing),
## we opt to simply store everything in an ndarray. If more data is present
## we should probably use the `tf.data` API.

import numpy as np
from os import path
from skimage import img_as_float
from skimage.io import imread
from skimage.transform import resize
from pycocotools.coco import COCO

def load_data(data_dir, data_type, image_shape=(224,224), sigma=1.0, num_input=None):
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
    image_ids = coco.getImgIds(
        catIds=coco.getCatIds(catNms=['person'])
    ) # only load person images
    image_ids = np.random.permutation(image_ids)
    if num_input != None:
        image_ids = image_ids[:num_input]
    images = coco.loadImgs(image_ids)

    # Allocate feature and label tensor
    X = np.ndarray((len(images), image_shape[0], image_shape[1], 3))
    Y = np.zeros((len(images), image_shape[0], image_shape[1]))

    # Build the tensors
    for i, img_data in enumerate(images):
        # Build X (feature; scaled and resized input image)
        img_path = path.join(image_dir, img_data['file_name'])
        img = img_as_float(imread(img_path))
        scale = (img.shape[0] / image_shape[0], img.shape[1] / image_shape[1])
        X[i,:,:,:] = resize(img, image_shape, mode='reflect', anti_aliasing=True)

        # Build Y (label; ground truth confidence map)
        annos = coco.loadAnns(coco.getAnnIds(imgIds=img_data['image_id']))
        for anno in annos: # each annotation corresponds to a different person
            update_confidence_map(Y[i,:,:], anno, scale, sigma)

    return X, Y
    
def update_confidence_map(Y, anno, scale, sigma):
    n = anno["num_keypoints"]
    keypoints = anno["keypoints"]
    
    # Construct the individual confidence map
    Ytmp = np.zeros_like(Y)
    for k in range(n):
        x, y, visibility = keypoints[3*k], keypoints[3*k+1], keypoints[3*k+2]
        if visibility == 2: # labeled and visible
            # Scale the coordinate
            x = x / scale[0]
            y = y / scale[1]
            # Apply Gaussian kernel
            # TODO: vectorize this part
            for i in range(Ytmp.shape[0]):
                for j in range(Ytmp.shape[1]):
                    Ytmp[i,j] = Ytmp[i,j] + np.exp(-((i - x)**2 + (j - y)**2) / sigma**2)
    
    # Take the maximum
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = max(Y[i,j], Ytmp[i,j])
