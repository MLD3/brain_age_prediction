import numpy as np

def augmentImage(image):
    imageRot90 = np.rot90(image, axes=(0,1))
    imageRot180 = np.rot90(imageRot90, axes=(0,1))
    imageRot270 = np.rot90(imageRot180, axes=(0,1))
    imageFlipVert = np.flip(image, axis=0)
    imageFlipHor = np.flip(image, axis=1)
    return np.array([image, imageRot90, imageRot180, imageRot270, imageFlipVert, imageFlipHor])
