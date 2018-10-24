# Keypoints and skeletons lookup table
# person_cat = coco.loadCats(coco.getCatIds(catNms=['person']))[0]
# KEYPOINTS = person_cat['keypoints']
# SKELETONS = [(i-1, j-1) for i, j in person_cat['skeleton']] # convert to 0-based indexing

KEYPOINTS = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
             'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
             'left_knee','right_knee','left_ankle','right_ankle']
NUM_KEYPOINTS = len(KEYPOINTS)
SKELETONS = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
             (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

# Hyperparameters
image_shape = (224, 224) # standardized input shape
sigma = 1.0 # spread of the ground truth confidence map
learning_rate = 0.01
momentum = 0.9
