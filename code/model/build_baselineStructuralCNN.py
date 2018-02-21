import tensorflow as tf
import numpy as np
from utils.config import get
from utils.patches import ExtractImagePatches3D
from placeholders.shared_placeholders import *

def standardBatchNorm(inputs, trainingPL, momentum=0.9, name=None):
    return tf.layers.batch_normalization(inputs, training=trainingPL, momentum=momentum, name=name, reuse=tf.AUTO_REUSE)

def standardPool(inputs, kernel_size=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name=None):
    return tf.nn.max_pool3d(inputs, ksize=kernel_size, strides=strides, padding=padding, name=name)

def standardConvolution(inputs, filters, kernel_size=(3,3,3), activation=tf.nn.elu, strides=(1,1,1), padding='SAME', name=None):
    return tf.layers.conv3d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation=activation,
                            use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer(), name=name, reuse=tf.AUTO_REUSE)

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

def standardBlock(inputs, 
                  trainingPL, 
                  blockNumber, 
                  filters, 
                  kernelSize=(3,3,3),
                  normalize=True,
                  pool=True):
    with tf.variable_scope('ConvBlock{}'.format(blockNumber)):
        BlockConvolution1 = standardConvolution(inputs,
                                                filters=filters,
                                                name='Block{}Convolution1'.format(blockNumber),
                                                kernel_size=kernelSize)
        BlockConvolution2 = standardConvolution(BlockConvolution1,
                                                filters=filters,
                                                name='Block{}Convolution2'.format(blockNumber),
                                                kernel_size=kernelSize)
        outputLayer = BlockConvolution2
        
        if normalize:
            outputLayer = standardBatchNorm(outputLayer, trainingPL, name='Block{}BatchNorm'.format(blockNumber))
        if pool:
            outputLayer = standardPool(outputLayer, name='Block{}MaxPool'.format(blockNumber))
        return outputLayer

def attentionMap(inputs):
    with tf.variable_scope('attentionMap'):
        weightShape = inputs.shape.as_list()
        weightShape[0] = 1
        attentionWeight = tf.get_variable(tf.ones(shape=(weightShape), name='attentionWeight'))
        return tf.multiply(inputs, attentionWeight)

def baselineStructuralCNN(imagesPL,
                          trainingPL,
                          keepProbability=0.6):
    with tf.variable_scope('ConvolutionalNetwork'):
        if imagesPL.dtype != tf.float32:
            imagesPL = tf.cast(imagesPL, tf.float32, name='CastInputToFloat32')

        Block1 = standardBlock(imagesPL, trainingPL, blockNumber=1, filters=8)
        Block2 = standardBlock(Block1, trainingPL, blockNumber=2, filters=16)
        Block3 = standardBlock(Block2, trainingPL, blockNumber=3, filters=32)
        Block4 = standardBlock(Block3, trainingPL, blockNumber=4, filters=64)
        
        with tf.variable_scope('FullyConnectedLayers'):
            flattenedLayer = tf.layers.flatten(Block4)
            hiddenLayer = standardDense(inputs=flattenedLayer, units=256, name='hiddenLayer')
            droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=keepProbability, is_training=trainingPL)
            outputLayer = standardDense(droppedOutHiddenLayer, units=1, activation=None, use_bias=False, name='outputLayer')
        return outputLayer

def constantBaseline(imagesPL,
                          trainingPL,
                          keepProbability=0.6):
    with tf.variable_scope('ConvolutionalNetwork'):
        if imagesPL.dtype != tf.float32:
            imagesPL = tf.cast(imagesPL, tf.float32, name='CastInputToFloat32')

        Block1 = standardBlock(imagesPL, trainingPL, blockNumber=1, filters=64)
        Block2 = standardBlock(Block1, trainingPL, blockNumber=2, filters=64)
        Block3 = standardBlock(Block2, trainingPL, blockNumber=3, filters=64)
        Block4 = standardBlock(Block3, trainingPL, blockNumber=4, filters=64)
        
        with tf.variable_scope('FullyConnectedLayers'):
            flattenedLayer = tf.layers.flatten(Block4)
            hiddenLayer = standardDense(inputs=flattenedLayer, units=256, name='hiddenLayer')
            droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=keepProbability, is_training=trainingPL)
            outputLayer = standardDense(droppedOutHiddenLayer, units=1, activation=None, use_bias=False, name='outputLayer')
        return outputLayer
    
def reverseBaseline(imagesPL,
                          trainingPL,
                          keepProbability=0.6):
    with tf.variable_scope('ConvolutionalNetwork'):
        if imagesPL.dtype != tf.float32:
            imagesPL = tf.cast(imagesPL, tf.float32, name='CastInputToFloat32')

        Block1 = standardBlock(imagesPL, trainingPL, blockNumber=1, filters=64)
        Block2 = standardBlock(Block1, trainingPL, blockNumber=2, filters=32)
        Block3 = standardBlock(Block2, trainingPL, blockNumber=3, filters=16)
        Block4 = standardBlock(Block3, trainingPL, blockNumber=4, filters=8)
        
        with tf.variable_scope('FullyConnectedLayers'):
            flattenedLayer = tf.layers.flatten(Block4)
            hiddenLayer = standardDense(inputs=flattenedLayer, units=256, name='hiddenLayer')
            droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=keepProbability, is_training=trainingPL)
            outputLayer = standardDense(droppedOutHiddenLayer, units=1, activation=None, use_bias=False, name='outputLayer')
        return outputLayer

def depthPatchCNN(imagesPL,
                  trainingPL,
                  keepProbability=0.6,
                  strideSize=10):
    with tf.variable_scope('depthPatchCNN'):
        if imagesPL.dtype != tf.float32:
            imagesPL = tf.cast(imagesPL, tf.float32, name='CastInputToFloat32')
        
        with tf.variable_scope('PatchExtraction'):
            imagePatches = ExtractImagePatches3D(imagesPL, strideSize=strideSize)
        
        Block1 = standardBlock(imagePatches, trainingPL, blockNumber=1, filters=8)
        Block2 = standardBlock(Block1, trainingPL, blockNumber=2, filters=16)
        Block3 = standardBlock(Block2, trainingPL, blockNumber=3, filters=32)
        Block4 = standardBlock(Block3, trainingPL, blockNumber=4, filters=64)

        with tf.variable_scope('FullyConnectedLayers'):
            flattenedLayer = tf.layers.flatten(Block4)
            hiddenLayer = standardDense(inputs=flattenedLayer, units=256, name='hiddenLayer')
            droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=keepProbability, is_training=trainingPL)
            outputLayer = standardDense(droppedOutHiddenLayer, units=1, activation=None, use_bias=False, name='outputLayer')
        return outputLayer

def simpleCNN(imagesPL,
              trainingPL):
    with tf.variable_scope('simpleCNN'):
        if imagesPL.dtype != tf.float32:
            imagesPL = tf.cast(imagesPL, tf.float32, name='CastInputToFloat32')
        
        Block1 = standardBlock(imagesPL, trainingPL, blockNumber=1, filters=8)
        Block2 = standardBlock(Block1, trainingPL, blockNumber=2, filters=16)
        Block3 = standardBlock(Block2, trainingPL, blockNumber=3, filters=32)
        
        with tf.variable_scope('FullyConnectedLayers'):
            flattenedLayer = tf.layers.flatten(Block3)
            outputLayer = standardDense(inputs=flattenedLayer,
                                        units=1,
                                        activation=None,
                                        use_bias=False,
                                        name='outputLayer')
        return outputLayer
    
def reverseDepthCNN(imagesPL,
                    trainingPL,
                    keepProbability=0.6,
                    strideSize=10):
    with tf.variable_scope('depthPatchCNN'):
        if imagesPL.dtype != tf.float32:
            imagesPL = tf.cast(imagesPL, tf.float32, name='CastInputToFloat32')
        
        with tf.variable_scope('PatchExtraction'):
            imagePatches = ExtractImagePatches3D(imagesPL, strideSize=strideSize)
        
        Block1 = standardBlock(imagePatches, trainingPL, blockNumber=1, filters=64)
        Block2 = standardBlock(Block1, trainingPL, blockNumber=2, filters=32)
        Block3 = standardBlock(Block2, trainingPL, blockNumber=3, filters=16)
        Block4 = standardBlock(Block3, trainingPL, blockNumber=4, filters=8)

        with tf.variable_scope('FullyConnectedLayers'):
            flattenedLayer = tf.layers.flatten(Block4)
            hiddenLayer = standardDense(inputs=flattenedLayer, units=256, name='hiddenLayer')
            droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=keepProbability, is_training=trainingPL)
            outputLayer = standardDense(droppedOutHiddenLayer, units=1, activation=None, use_bias=False, name='outputLayer')
        return outputLayer

def constantDepthCNN(imagesPL,
                    trainingPL,
                    keepProbability=0.6,
                    strideSize=10):
    with tf.variable_scope('depthPatchCNN'):
        if imagesPL.dtype != tf.float32:
            imagesPL = tf.cast(imagesPL, tf.float32, name='CastInputToFloat32')
        
        with tf.variable_scope('PatchExtraction'):
            imagePatches = ExtractImagePatches3D(imagesPL, strideSize=strideSize)
        
        Block1 = standardBlock(imagePatches, trainingPL, blockNumber=1, filters=64)
        Block2 = standardBlock(Block1, trainingPL, blockNumber=2, filters=64)
        Block3 = standardBlock(Block2, trainingPL, blockNumber=3, filters=64)
        Block4 = standardBlock(Block3, trainingPL, blockNumber=4, filters=64)

        with tf.variable_scope('FullyConnectedLayers'):
            flattenedLayer = tf.layers.flatten(Block4)
            hiddenLayer = standardDense(inputs=flattenedLayer, units=256, name='hiddenLayer')
            droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=keepProbability, is_training=trainingPL)
            outputLayer = standardDense(droppedOutHiddenLayer, units=1, activation=None, use_bias=False, name='outputLayer')
        return outputLayer