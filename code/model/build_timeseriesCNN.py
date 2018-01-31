import tensorflow as tf
import numpy as np
from utils.config import get

def convolution1D(inputs, filters, kernel_size=12, activation=tf.nn.elu, strides=1, name=None):
    return tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding='SAME', activation=activation,
                            use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer(), name=name, reuse=tf.AUTO_REUSE)

def pool1D(inputs, kernel_size=6, strides=6, padding='SAME', name=None):
    return tf.layers.max_pooling1d(inputs, pool_size=kernel_size, strides=strides, padding=padding, name=name)

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

def TimeseriesCNN(inputPL, trainingPL, keepProbability=0.6):
    with tf.variable_scope('TimeseriesCNN'):
        conv1 = convolution1D(inputs=inputPL,
                              filters=264,
                              name='conv1')
        pool1 = pool1D(inputs=conv1,
                       name='pool1')
        conv2 = convolution1D(inputs=pool1,
                              filters=128,
                              kernel_size=5,
                              name='conv2')
        pool2 = pool1D(inputs=conv2,
                       kernel_size=5,
                       strides=5,
                       name='pool2')
        flattenedLayer = tf.layers.flatten(pool2)
        fullyConnectedLayer = standardDense(flattenedLayer, units=128, name='fullyConnectedLayer')
        droppedOutLayer = tf.contrib.layers.dropout(inputs=fullyConnectedLayer, keep_prob=keepProbability, is_training=trainingPL)
        outputLayer = standardDense(droppedOutLayer, units=1, activation=None, use_bias=False, name='outputLayer')
        return outputLayer
