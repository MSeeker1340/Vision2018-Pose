# Vision2018-Pose
Class project repository for CSCI-GA.3033-012: Vision Meets Machine Learning (Fall 2018)

We explore and experiment the methods and processes of real-time 2D multiperson pose estimation described in the paper: https://arxiv.org/pdf/1611.08050.pdf

Previous results and references:
1. Original version: https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation
2. OpenPose (C++ version): https://github.com/CMU-Perceptual-Computing-Lab/openpose
3. OpenPose Plus (Latest version using TensorFlow, OpenPose and TensorRT): https://github.com/tensorlayer/openpose-plus

## Dowload and Unzip Data
get_data.sh

## Preprocessing 
src/preprocessing.py
1. Load local data using COCO protocol. 
2. Resize each image to X, an array of images (s1,s2,3), where (s1,s2) is the image_shape parameter (default s1=s2=224). 
We require that s is divisible by 8.
3. COCO annotations label the body joints(src/config.py KEYPOINTS) already, and we use these points to build confidence map of each image and store in Y. Y is an array of (s/8, s/8, NUM_KEYPOINTS).
The confidence map is calculated using a Gaussian kernel and take max

## Plot and Examine Previous Results
src/plot_result.py.ipynb
