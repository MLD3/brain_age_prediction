import tensorflow as tf
import numpy as np
from utils.config import get
from placeholders.shared_placeholders import *

def standardBatchNorm(inputs, trainingPL, momentum=0.9):
    return tf.layers.batch_normalization(inputs, training=trainingPL, momentum=momentum)

def standardConvolution(inputs, filters, kernel_size, activation=tf.nn.elu, strides=(1,1), name=None):
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding='valid', activation=activation,
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

def baselineROICNN(matricesPL, trainingPL, keepProbability=get('TRAIN.ROI_BASELINE.KEEP_PROB'), firstHiddenLayerUnits=96, secondHiddenLayerUnits=0, defaultActivation=tf.nn.elu):
    matrixDim = get('DATA.MATRICES.DIMENSION')
    filtersInFirstConvolution = 64
    filtersInSecondConvolution = 128
    numberOfUnitsInOutputLayer = 1

    kernelSizeOfFirstConvolution = [1, matrixDim]
    kernelSizeOfSecondConvolution = [matrixDim, 1]

    dropoutKeepProbability = keepProbability

    ########## FIRST CONVOLUTIONAL LAYER: (1, matrixDim)  ##########
    rowConvolution = standardConvolution(inputs=matricesPL, filters=filtersInFirstConvolution, activation=defaultActivation,
                                        kernel_size=kernelSizeOfFirstConvolution, name='rowConvolution')
    rowNormalized = standardBatchNorm(inputs=rowConvolution, trainingPL=trainingPL)

    ########## SECOND CONVOLUTIONAL LAYER: (matrixDim, 1) ##########
    columnConvolution = standardConvolution(inputs=rowNormalized, filters=filtersInSecondConvolution, activation=defaultActivation,
                                         kernel_size=kernelSizeOfSecondConvolution, name='columnConvolution')
    columnNormalized = standardBatchNorm(inputs=columnConvolution, trainingPL=trainingPL)

    ########## FIRST FULLY CONNECTED HIDDEN LAYER: firstHiddenLayerUnits units     ##########
    flattenedConvolution = tf.contrib.layers.flatten(columnNormalized)
    hiddenLayer = standardDense(inputs=flattenedConvolution, units=firstHiddenLayerUnits, activation=defaultActivation, name='firstHiddenLayer')
    droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=dropoutKeepProbability, is_training=trainingPL)

    ########## OPTIONAL: SECOND FULLY CONNECTED HIDDEN LAYER: secondHiddenLayerUnits units     ##########
    if secondHiddenLayerUnits > 0:
        hiddenLayer = standardDense(inputs=droppedOutHiddenLayer, units=secondHiddenLayerUnits, activation=defaultActivation, name='secondHiddenLayer')
        droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=dropoutKeepProbability, is_training=trainingPL)

    ########## OUTPUT LAYER: 1 UNIT (REGRESSION)         ##########
    outputLayer = standardDense(inputs=droppedOutHiddenLayer, units=numberOfUnitsInOutputLayer, activation=None, use_bias=False, name='outputLayer')
    return outputLayer

if __name__ == '__main__':
    matricesPL, labelsPL = MatrixPlaceholders()
    trainingPL = TrainingPlaceholder()
    outputLayer = baselineROICNN(matricesPL, trainingPL)
