import tensorflow as tf
import numpy as np
from utils import get


def baselineROICNN(matricesPL, trainingPL):
    matrixDim = get('DATA.MATRICES.DIMENSION')
    filtersInFirstConvolution = 64
    filtersInSecondConvolution = 128
    numberOfUnitsInHiddenLayer = 96
    numberOfUnitsInOutputLayer = 1

    dropoutKeepProbability = 0.6

    rowCovolution = tf.layers.conv2d(inputs=matricesPL, filters=filtersInFirstConvolution, kernel_size=(1, matrixDim),
                                     strides=(1,1), padding='SAME', activation=tf.nn.elu,
                                     use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.zeros_initializer(), name='rowConvolution')
    rowNormalized = tf.layers.batch_normalization(rowConvolution)
    columnConvolution = tf.layers.conv2d(inputs=rowNormalized, filters=filtersInSecondConvolution, kernel_size=(matrixDim, 1)
                                     strides=(1,1), padding='SAME', activation=tf.nn.elu,
                                     use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.zeros_initializer(), name='columnConvolution')
    columnNormalized = tf.layers.batch_normalization(columnConvolution)
    flattenedConvolution = tf.contrib.layers.flatten(columnNormalized)
    hiddenLayer = tf.layers.dense(inputs=flattenedConvolution, units=numberOfUnitsInHiddenLayer, activation=tf.nn.elu,
                                     bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='hiddenLayer')
    droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=dropoutKeepProbability, is_training=trainingPL)
    outputLayer = tf.layers.dense(input=droppedOutHiddenLayer, units=numberOfUnitsInOutputLayer, activation=None,
                                     use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='outputLayer')
    return outputLayer
