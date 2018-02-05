import numpy as np
import tensorflow as tf

def ExtractImagePatches3D(images, strideSize):
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
                                    rowIndex-strideSize:rowIndex,
                                    colIndex-strideSize:colIndex,
                                    depthIndex-strideSize:depthIndex,
                                    :]
                patches.append(imageSlice)
                depthIndex += strideSize
            colIndex += strideSize
        rowIndex += strideSize
    imagePatches = tf.concat(patches, axis=4)
    return imagePatches
