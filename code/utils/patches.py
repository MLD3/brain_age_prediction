import numpy as np
import tensorflow as tf

def ExtractImagePatchesDEPRECATED(images, strideSize, kernelSize=3):
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
    rowIndex = strideSize + kernelSize
    while rowIndex <= numRows:
        colIndex = strideSize + kernelSize
        while colIndex <= numCols:
            depthIndex = strideSize + kernelSize
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

def ExtractImagePatches3D(images, scale=2, kernelSize=3, sliceIndex=None, align=False):
    if scale == 1:
        return images
    _, numRows, numCols, depth, _ = images.get_shape().as_list()
    patches = []
    rowStride = int((numRows - 2 * kernelSize) / scale)
    colStride = int((numCols - 2 * kernelSize) / scale)
    depthStride = int((depth - 2 * kernelSize) / scale)
    runningIndex = 0
    X = 1
    Y = 2
    Z = 3
    alignAxes=[None, [Z], [Y], [Y, Z], [X], [X, Z], [X, Y], [X, Y, Z]]
    
    rowIndex = rowStride + kernelSize
    while rowIndex <= numRows:
        colIndex = colStride + kernelSize
        while colIndex <= numCols:
            depthIndex = depthStride + kernelSize
            while depthIndex <= depth:
                # Extract an image slice of size
                # [batchSize, rowStride, colStride, depthStride, numChannels]
                imageSlice = images[:,
                                    max(rowIndex-rowStride-kernelSize, 0):min(rowIndex+kernelSize, numRows),
                                    max(colIndex-colStride-kernelSize, 0):min(colIndex+kernelSize, numCols),
                                    max(depthIndex-depthStride-kernelSize, 0):min(depthIndex+kernelSize, depth),
                                    :]
                if sliceIndex is not None and sliceIndex == runningIndex:
                    return imageSlice
                if align and alignAxes[runningIndex] is not None:
                    imageSlice = tf.reverse(imageSlice, axis=alignAxes[runningIndex])
                
                runningIndex += 1
                patches.append(imageSlice)
                depthIndex += depthStride
            colIndex += colStride
        rowIndex += rowStride
    imagePatches = tf.concat(patches, axis=4)
    return imagePatches