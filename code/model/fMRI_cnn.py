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

def pool_layer(x, input_height, input_width, input_depth, input_channels, k_length, stride, padding):
    x = tf.reshape(x, [-1, input_height, input_width, input_depth, input_channels])
    return tf.nn.max_pool3d(x, ksize=[1, k_length, k_length, k_length, 1], strides=[1, stride, stride, stride, 1], padding=padding)


def weight_var(shape, mean, sd):
    return tf.Variable(tf.truncated_normal(shape, mean=mean, stddev=sd))

def bias_var(shape, value):
    return tf.Variable(tf.constant(value, shape=shape)) 

def conv_layer(x, input_height, input_width, input_depth, 
                input_channels, filter_size, output_channels, 
                padding, act_type, mean, sd, bias, stride):
    x = tf.reshape(x, [-1, input_height, input_width, input_depth, input_channels])
    filter_W = weight_var([filter_size, filter_size, filter_size, input_channels, output_channels], mean, sd)
    bias_W = bias_var([output_channels], bias)

    conv = tf.nn.conv3d(x, filter = filter_W, strides=[1, stride, stride, stride, 1], padding=padding)
    return activation(conv + bias_W, type=act_type)

def fully_connected_layer(x, shape, act_type, mean, sd, bias):
    x = tf.reshape(x, [-1, shape[0]])
    fc_W = weight_var(shape, mean, sd)
    fc_B = bias_var([shape[-1]], bias)

    return activation(tf.matmul(x, fc_W) + fc_B, type=act_type)


def fMRI_4D_CNN(meanDefault=0.0, sdDefault=0.01, biasDefault=0.1):
    input_layer = tf.placeholder(tf.float32, shape=[None, 52*62*45])
    # 52*62*45 -> 52*62*45*2
    conv1 = conv_layer(x = input_layer, input_height = 52, input_width = 62, 
                        input_depth = 45, input_channels = 1, filter_size = 7, 
                        output_channels=2, padding='SAME', act_type='relu', 
                        mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=1)
    # 52*62*45*2 -> 26*31*23*2
    pool1 = pool_layer(x=conv1, input_height = 52, input_width = 62, input_depth = 45, input_channels=2, k_length=2, stride=2, padding='SAME')
    # 26*31*23*2 -> 26*31*23*8
    conv2 = conv_layer(x=pool1, input_height = 26, input_width = 31, input_depth = 23, input_channels=2,
                        filter_size=5, output_channels=8, padding='SAME',
                        act_type='relu', mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=1)
    # 26*31*23*8 -> 13*16*12*8
    pool2 = pool_layer(x=conv2, input_height = 26, input_width = 31, input_depth = 23, input_channels=8, k_length=2, stride=2, padding='SAME')
    # 13*16*12*8 -> 13*16*12*16
    conv3 = conv_layer(x=pool2, input_height = 13, input_width = 16, input_depth = 12, input_channels=8,
                        filter_size=3, output_channels=16, padding='SAME',
                        act_type='relu', mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=1)
    # 13*16*12*16 -> 7*8*6*16
    pool3 = pool_layer(x=conv3, input_height = 13, input_width = 16, input_depth = 12, input_channels=16, k_length=3, stride=2, padding='SAME')
    # fully connected
    fn1 = fully_connected_layer(pool3, shape=[7 * 8 * 6 * 16, 1], act_type='none', mean=meanDefault, sd=sdDefault, bias=biasDefault)
    return input_layer, fn1
