import tensorflow as tf
import numpy as np
from utils.config import get
from placeholders.shared_placeholders import *

def standardBatchNorm(inputs, trainingPL, momentum=0.9):
    return tf.layers.batch_normalization(inputs, training=trainingPL, momentum=momentum)

def standardConvolution(inputs, filters, kernel_size, strides=(1,1), name=None):
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding='valid', activation=tf.nn.elu,
                            use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer(), name=name)

def standardDense(inputs, units, activation=tf.nn.elu, use_bias=True, name=None):
    if use_bias:
        return tf.layers.dense(inputs=inputs, units=units, activation=activation,
                           bias_initializer=tf.zeros_initializer(),
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           name=name)
    else:
        return tf.layers.dense(inputs=inputs, units=units, activation=activation,
                           use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           bias_initializer=tf.zeros_initializer(), name=name)

def baselineROICNN(matricesPL, trainingPL):
    matrixDim = get('DATA.MATRICES.DIMENSION')
    filtersInFirstConvolution = 64
    filtersInSecondConvolution = 128
    numberOfUnitsInHiddenLayer = 96
    numberOfUnitsInOutputLayer = 1

    kernelSizeOfFirstConvolution = [1, matrixDim]
    kernelSizeOfSecondConvolution = [matrixDim, 1]

    dropoutKeepProbability = 0.6

    ########## FIRST CONVOLUTIONAL LAYER: (1, matrixDim)  ##########
    rowConvolution = standardConvolution(inputs=matricesPL, filters=filtersInFirstConvolution,
                                        kernel_size=kernelSizeOfFirstConvolution, name='rowConvolution')
    rowNormalized = standardBatchNorm(inputs=rowConvolution, trainingPL=trainingPL)

    ########## SECOND CONVOLUTIONAL LAYER: (matrixDim, 1) ##########
    columnConvolution = standardConvolution(inputs=rowNormalized, filters=filtersInSecondConvolution,
                                         kernel_size=kernelSizeOfSecondConvolution, name='columnConvolution')
    columnNormalized = standardBatchNorm(inputs=columnConvolution, trainingPL=trainingPL)

    ########## FULLY CONNECTED HIDDEN LAYER: 96 units     ##########
    flattenedConvolution = tf.contrib.layers.flatten(columnNormalized)
    hiddenLayer = standardDense(inputs=flattenedConvolution, units=numberOfUnitsInHiddenLayer, name='hiddenLayer')
    droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=dropoutKeepProbability, is_training=trainingPL)

    ########## OUTPUT LAYER: 1 UNIT (REGRESSION)         ##########
    outputLayer = standardDense(inputs=droppedOutHiddenLayer, units=numberOfUnitsInOutputLayer, activation=None, use_bias=False, name='outputLayer')
    return outputLayer

if __name__ == '__main__':
    matricesPL, labelsPL = MatrixPlaceholders()
    trainingPL = TrainingPlaceholder()
    outputLayer = baselineROICNN(matricesPL, trainingPL)
