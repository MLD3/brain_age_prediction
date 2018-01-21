import tensorflow as tf
import numpy as np
from utils.config import get
from placeholders.shared_placeholders import *

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

def baselineLSTM(timecoursePL):
    numFinalDense = 1
    numHiddenUnits = 64

    with tf.variable_scope('RecurrentLayer'):
        lstmCell = tf.contrib.rnn.BasicLSTMCell(num_units=numHiddenUnits, activation=tf.nn.elu)
        outputs, states = tf.nn.dynamic_rnn(lstmCell, timecoursePL, dtype=tf.float32)

    with tf.variable_scope('DenseLayer'):
        outputLayer = standardDense(outputs[:, -1, :], units=numFinalDense, activation=None, use_bias=False)

    return outputLayer
