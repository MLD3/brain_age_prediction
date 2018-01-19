import tensorflow as tf
import numpy as np
from utils.config import get
from placeholders.shared_placeholders import *

def standardBatchNorm(inputs, trainingPL, momentum=0.9, name=None):
    return tf.layers.batch_normalization(inputs, training=trainingPL, momentum=momentum, name=name)

def standardPool(inputs, kernel_size=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name=None):
    return tf.nn.max_pool3d(inputs, ksize=kernel_size, strides=strides, padding=padding, name=name)

def standardConvolution(inputs, filters, kernel_size=(3,3,3), activation=tf.nn.elu, strides=(1,1,1), padding='SAME', name=None):
    return tf.layers.conv3d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation=activation,
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

def standardBlock(inputs, trainingPL, blockNumber, filters):
    with tf.variable_scope('ConvBlock{}'.format(blockNumber)):
        #### 3x3x3 Convolution ####
        BlockConvolution1 = standardConvolution(inputs, filters=filters, name='Block{}Convolution1'.format(blockNumber))
        #### 3x3x3 Convolution ####
        BlockConvolution2 = standardConvolution(BlockConvolution1, filters=filters, name='Block{}Convolution2'.format(blockNumber))
        #### Batch Normalization ####
        BlockBatchNorm = standardBatchNorm(BlockConvolution2, trainingPL, name='Block{}BatchNorm'.format(blockNumber))
        #### Max Pooling ####
        BlockMaxPool = standardPool(BlockBatchNorm, name='Block{}MaxPool'.format(blockNumber))
        return BlockMaxPool

def attentionMap(inputs):
    with tf.variable_scope('attentionMap'):
        weightShape = list(inputs.shape)
        weightShape[0] = 1
        attentionWeight = tf.Variable(tf.ones(shape=weightShape, name='attentionWeight'))
        return tf.multiply(inputs, attentionWeight)

def baselineStructuralCNN(imagesPL, trainingPL, keepProbability=get('TRAIN.ROI_BASELINE.KEEP_PROB'), defaultActivation=tf.nn.elu, optionalHiddenLayerUnits=0, useAttentionMap=False):
    if useAttentionMap:
        imagesPL = attentionMap(imagesPL)

    ################## FIRST BLOCK ##################
    Block1 = standardBlock(imagesPL, trainingPL, blockNumber=1, filters=8)

    ################## SECOND BLOCK ##################
    Block2 = standardBlock(Block1, trainingPL, blockNumber=2, filters=8)

    ################## THIRD BLOCK ##################
    Block3 = standardBlock(Block2, trainingPL, blockNumber=3, filters=8)

    ################## FOURTH BLOCK ##################
    Block4 = standardBlock(Block3, trainingPL, blockNumber=4, filters=8)

    ################## FIFTH BLOCK ##################
    Block5 = standardBlock(Block4, trainingPL, blockNumber=5, filters=8)

    with tf.variable_scope('FullyConnectedLayers'):
        flattenedLayer = tf.layers.flatten(Block5)
        if optionalHiddenLayerUnits > 0:
            optionalHiddenLayer = standardDense(inputs=flattenedLayer, units=optionalHiddenLayerUnits, activation=defaultActivation, name='optionalHiddenLayer')
            droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=optionalHiddenLayer, keep_prob=keepProbability, is_training=trainingPL)
            flattenedLayer = droppedOutHiddenLayer

        numberOfUnitsInOutputLayer = 1
        outputLayer = standardDense(flattenedLayer, units=numberOfUnitsInOutputLayer, activation=None, use_bias=False, name='outputLayer')
    return outputLayer
