import tensorflow as tf
import numpy as np

def activation(x, type):
    if type == 'none' or type == 'linear':
        return x
    if type == 'relu':
        return tf.nn.relu(x)
    if type == 'tanh':
        return tf.tanh(x)
    if type == 'sigmoid':
        return tf.sigmoid(x)

def weight_var(shape, mean, sd):
    return tf.Variable(tf.truncated_normal(shape, mean=mean, stddev=sd))

def bias_var(shape, value):
    return tf.Variable(tf.constant(value, shape=shape))

def conv_layer(x, input_size, input_channels, filter_size, output_channels, padding, act_type, mean, sd, bias, stride):
    x = tf.reshape(x, [-1, input_size, input_size, input_channels])
    filter_W = weight_var([filter_size, filter_size, input_channels, output_channels], mean, sd)
    bias_W = bias_var([output_channels], bias)

    conv = tf.nn.conv2d(x, filter=filter_W, strides=[1, stride, stride, 1], padding=padding)
    return activation(conv + bias_W, type=act_type)

def fully_connected_layer(x, shape, act_type, mean, sd, bias):
    x = tf.reshape(x, [-1, shape[0]])
    fc_W = weight_var(shape, mean, sd)
    fc_B = bias_var([shape[-1]], bias)

    return activation(tf.matmul(x, fc_W) + fc_B, type=act_type)

def pool_layer(x, input_size, input_channels, k_length, stride, padding):
    x = tf.reshape(x, [-1, input_size, input_size, input_channels])
    return tf.nn.max_pool(x, ksize=[1, k_length, k_length, 1], strides=[1, stride, stride, 1], padding=padding)

def cnn():
    ##########################################################################
    ####################### Define custom architecture #######################
    ##########################################################################
