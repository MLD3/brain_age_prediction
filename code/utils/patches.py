import numpy as np
import tensorflow as tf

def ExtractImagePatches3D(images, strideSize, kernelSize=3):
    """
        images: a 5D tensor of shape (batchSize, numRows, numCols, depth, numChannels)
        returns: a 5D tensor of shape (batchSize, strideSize, strideSize, strideSize, numChannels * numPatches)
        where numPatches is the number of strideSize * strideSize * strideSize patches that fit
        in the original image

        This function uses no padding. If strideSize does not divide evenly
        into the dimensions of the images, the edges of the images along
        the non-divisible dimensions will be clipped by the remainder.
    """
    _, numRows, numCols, depth, _ = images.get_shape().as_list()
    patches = []
    rowIndex = strideSize
    while rowIndex <= numRows:
        colIndex = strideSize
        while colIndex <= numCols:
            depthIndex = strideSize
            while depthIndex <= depth:
                # Extract an image slice of size
                # [batchSize, strideSize, strideSize, strideSize, numChannels]
                imageSlice = images[:,
                                    max(rowIndex-strideSize-kernelSize, 0):min(rowIndex+kernelSize, numRows),
                                    max(colIndex-strideSize-kernelSize, 0):min(colIndex+kernelSize, numCols),
                                    max(depthIndex-strideSize-kernelSize, 0):min(depthIndex+kernelSize, depth),
                                    :]
                patches.append(imageSlice)
                depthIndex += strideSize
            colIndex += strideSize
        rowIndex += strideSize
    imagePatches = tf.concat(patches, axis=4)
    return imagePatches
