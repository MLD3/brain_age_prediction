import numpy as np
import tensorflow as tf

def pariwiseDice(patches):
    patches = (patches > 0).astype(np.float32)
    numberChannels = patches.shape[4]
    accumulatedDice = 0
    accumulatedCounter = 0
    for i in range(numberChannels - 1):
        for j in range(i + 1, numberChannels):
            numerator = 2 * np.sum(np.min(patches[:,:,:,:, [i, j]], axis=4))
            denom = np.sum(patches[:,:,:,:,i]) + np.sum(patches[:,:,:,:,j])
            diceCoeff = numerator / denom
            accumulatedDice += diceCoeff
            accumulatedCounter += 1
    averageDice = accumulatedDice / accumulatedCounter
    return averageDice

def pairwiseDiceTF(imagesPL):
    with tf.variable_scope('diceCoeff'):
        imagesPL = tf.cast((imagesPL > 0), tf.float32)
        batchSize, numRows, numCols, depth, numberChannels = imagesPL.get_shape().as_list()
        diceVar = tf.Variable(dtype=tf.float32, initial_value=0, name='diceNumerator', trainable=False)
        diceCounter = tf.Variable(dtype=tf.float32, initial_value=0, name='diceDenominator', trainable=False)
        numAccum = tf.constant(shape=(), dtype=tf.float32, value=0)
        denAccum = tf.constant(shape=(), dtype=tf.float32, value=0)
        for i in range(numberChannels - 1):
            for j in range(i + 1, numberChannels):
                numerators = 2 * tf.reduce_sum(tf.reduce_min(tf.stack([imagesPL[:,:,:,:,i], imagesPL[:,:,:,:,j]], axis=4), axis=4), axis=[1,2,3])
                denoms = tf.reduce_sum(imagesPL[:, :, :, :, i], axis=[1,2,3]) + tf.reduce_sum(imagesPL[:, :, :, :, j], axis=[1,2,3])
                diceCoeff = tf.reduce_sum(numerators / denoms)
                numberElements = tf.cast(tf.shape(denoms)[0], tf.float32)
                numAccum = numAccum + diceCoeff
                denAccum = denAccum + numberElements
        updateOp = tf.group(tf.assign_add(diceVar, numAccum), tf.assign_add(diceCounter, denAccum))
        valueOp = diceVar / diceCounter
    return valueOp, updateOp

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

def ExtractImagePatches3D(images, scale=2, kernelSize=3, sliceIndex=None, align=False, randomFlips=False):
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

    if randomFlips:
        randomAxes = []
        for i in range(8):
            randomAxes.append(tf.Variable(
            initial_value=tf.unique(tf.random_uniform(shape=tf.random_uniform(
                                                shape=(1,), minval=0, maxval=3, dtype=tf.int32, name='uniformShape_{}'.format(i)), 
                                            minval=1, maxval=4, dtype=tf.int32, name='uniformValue_{}'.format(i)))[0], 
                                 validate_shape=False, name='randomFlipAxis_{}'.format(i), trainable=False))
        
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
                if randomFlips:
                    imageSlice = tf.reverse(imageSlice, axis=randomAxes[runningIndex])
                patches.append(imageSlice)
                runningIndex += 1
                depthIndex += depthStride
            colIndex += colStride
        rowIndex += rowStride
    imagePatches = tf.concat(patches, axis=4)
    return imagePatches
