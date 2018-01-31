import tensorflow as tf
import numpy as np
from utils.config import get
from placeholders.shared_placeholders import *

def standardBatchNorm(inputs, trainingPL, momentum=0.9, name=None):
    return tf.layers.batch_normalization(inputs, training=trainingPL, momentum=momentum, name=name, reuse=tf.AUTO_REUSE)

def standardDense(inputs, units, activation=tf.nn.elu, use_bias=True, name=None):
    if use_bias:
        return tf.layers.dense(inputs=inputs, units=units, activation=activation,
                           bias_initializer=tf.zeros_initializer(),
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           name=name, reuse=tf.AUTO_REUSE)
    else:
        return tf.layers.dense(inputs=inputs, units=units, activation=activation,
                           use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           bias_initializer=tf.zeros_initializer(), name=name, reuse=tf.AUTO_REUSE)

def convolution2D(inputs, filters, kernel_size=(3,3), activation=tf.nn.elu, strides=(1,1), name=None):
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding='SAME', activation=activation,
                            use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer(), name=name, reuse=tf.AUTO_REUSE)

def pool2D(inputs, kernel_size=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=None):
    return tf.nn.max_pool(inputs, ksize=kernel_size, strides=strides, padding=padding, name=name)

def block2D(inputs, trainingPL, blockNumber, filters):
    with tf.variable_scope('2DConvBlock{}'.format(blockNumber)):
        #### 3x3x3 Convolution ####
        BlockConvolution1 = convolution2D(inputs, filters=filters, name='Block{}Convolution1'.format(blockNumber))
        #### 3x3x3 Convolution ####
        BlockConvolution2 = convolution2D(BlockConvolution1, filters=filters, name='Block{}Convolution2'.format(blockNumber))
        #### Batch Normalization ####
        BlockBatchNorm = standardBatchNorm(BlockConvolution2, trainingPL, name='Block{}BatchNorm'.format(blockNumber))
        #### Max Pooling ####
        BlockMaxPool = pool2D(BlockBatchNorm, name='Block{}MaxPool'.format(blockNumber))
        return BlockMaxPool

def SliceCNN(imagesPL, trainingPL, keepProbability=get('TRAIN.CNN_BASELINE.KEEP_PROB'), defaultActivation=tf.nn.elu, optionalHiddenLayerUnits=0, downscaleRate=None):
    with tf.variable_scope('ConvolutionalNetwork'):
        if imagesPL.dtype != tf.float32:
            imagesPL = tf.cast(imagesPL, tf.float32, name='CastInputToFloat32')

        if downscaleRate:
            if isinstance(downscaleRate, int):
                downscaleSize = [1, downscaleRate, downscaleRate, 1]
                imagesPL = pool2D(imagesPL, kernel_size=downscaleSize, strides=downscaleSize)
            elif isinstance(downscaleRate, (list, tuple)) and len(downscaleRate) == 2:
                downscaleSize = [1, downscaleRate[0], downscaleRate[1], 1]
                imagesPL = pool2D(imagesPL, kernel_size=downscaleSize, strides=downscaleSize)
            else:
                raise ValueError('Unrecognized downscale rate: {}'.format(downscaleRate))

        ################## FIRST BLOCK ##################
        Block1 = block2D(imagesPL, trainingPL, blockNumber=1, filters=8)

        ################## SECOND BLOCK ##################
        Block2 = block2D(Block1, trainingPL, blockNumber=2, filters=16)

        ################## THIRD BLOCK ##################
        Block3 = block2D(Block2, trainingPL, blockNumber=3, filters=32)

        Block4 = block2D(Block3, trainingPL, blockNumber=4, filters=64)

        with tf.variable_scope('FullyConnectedLayers'):
            flattenedLayer = tf.layers.flatten(Block4)
            if optionalHiddenLayerUnits > 0:
                optionalHiddenLayer = standardDense(inputs=flattenedLayer, units=optionalHiddenLayerUnits, activation=defaultActivation, name='optionalHiddenLayer')
                droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=optionalHiddenLayer, keep_prob=keepProbability, is_training=trainingPL)
                flattenedLayer = droppedOutHiddenLayer

            numberOfUnitsInOutputLayer = 1
            outputLayer = standardDense(flattenedLayer, units=numberOfUnitsInOutputLayer, activation=None, use_bias=False, name='outputLayer')
    return outputLayer
