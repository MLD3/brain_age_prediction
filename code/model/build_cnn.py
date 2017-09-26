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

def cnnDefault(meanDefault=0.0, sdDefault=0.01, biasDefault=0.1):
    ##########################################################################
    ####################### Define custom architecture #######################
    ##########################################################################
    input_layer = tf.placeholder(tf.float32, shape=[None, 264*264])
    conv1 = conv_layer(x=input_layer, input_size=264, input_channels=1,
                        filter_size=3, output_channels=8, padding='SAME',
                        act_type='relu', mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=1)
    pool1 = pool_layer(x=conv1, input_size=264, input_channels=8, k_length=3, stride=2, padding='SAME')
    conv2 = conv_layer(x=pool1, input_size=132, input_channels=8,
                        filter_size=3, output_channels=16, padding='SAME',
                        act_type='relu', mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=1)
    pool2 = pool_layer(x=conv2, input_size=132, input_channels=16, k_length=3, stride=2, padding='SAME')
    conv3 = conv_layer(x=pool2, input_size=66, input_channels=16,
                        filter_size=3, output_channels=32, padding='SAME',
                        act_type='relu', mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=1)
    pool3 = pool_layer(x=conv3, input_size=66, input_channels=32, k_length=3, stride=2, padding='SAME')
    conv4 = conv_layer(x=pool3, input_size=33, input_channels=32,
                        filter_size=3, output_channels=64, padding='SAME',
                        act_type='relu', mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=1)
    pool4 = pool_layer(x=conv4, input_size=33, input_channels=64, k_length=3, stride=2, padding='SAME')
    conv5 = conv_layer(x=pool4, input_size=17, input_channels=64,
                        filter_size=3, output_channels=128, padding='SAME',
                        act_type='relu', mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=1)
    pool5 = pool_layer(x=conv5, input_size=17, input_channels=128, k_length=3, stride=2, padding='SAME')
    fn1 = fully_connected_layer(pool5, shape=[9 * 9 * 128, 1], act_type='none', mean=meanDefault, sd=sdDefault, bias=biasDefault)
    return (input_layer, fn1)

def cnnSmall(meanDefault=0.0, sdDefault=0.01, biasDefault=0.1):
    ##########################################################################
    ####################### Define custom architecture #######################
    ##########################################################################
    input_layer = tf.placeholder(tf.float32, shape=[None, 264*264])
    conv1 = conv_layer(x=input_layer, input_size=264, input_channels=1,
                        filter_size=4, output_channels=8, padding='SAME',
                        act_type='relu', mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=2)
    pool1 = pool_layer(x=conv1, input_size=132, input_channels=8, k_length=3, stride=2, padding='SAME')
    conv2 = conv_layer(x=pool1, input_size=66, input_channels=8,
                        filter_size=4, output_channels=32, padding='SAME',
                        act_type='relu', mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=3)
    pool2 = pool_layer(x=conv2, input_size=22, input_channels=32, k_length=3, stride=2, padding='SAME')
    conv3 = conv_layer(x=pool2, input_size=11, input_channels=32,
                        filter_size=4, output_channels=64, padding='SAME',
                        act_type='relu', mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=1)
    pool3 = pool_layer(x=conv3, input_size=11, input_channels=64, k_length=3, stride=2, padding='SAME')
    fn1 = fully_connected_layer(pool3, shape=[6 * 6 * 64, 256], act_type='none', mean=meanDefault, sd=sdDefault, bias=biasDefault)
    fn2 = fully_connected_layer(fn1, shape=[256, 1], act_type='none', mean=meanDefault, sd=sdDefault, bias=biasDefault)
    return (input_layer, fn2)

def cnnNeural(meanDefault=0.0, sdDefault=0.01, biasDefault=0.1, act_type='relu'):
    input_layer = tf.placeholder(tf.float32, shape=[None, 264*264])
    fn1 = fully_connected_layer(pool3, shape=[264*264, 64*64], act_type=act_type, mean=meanDefault, sd=sdDefault, bias=biasDefault)
    fn2 = fully_connected_layer(fn1, shape=[64*64, 32*32], act_type=act_type, mean=meanDefault, sd=sdDefault, bias=biasDefault)
    fn3 = fully_connected_layer(pool3, shape=[32 * 32, 8*8], act_type=act_type, mean=meanDefault, sd=sdDefault, bias=biasDefault)
    fn4 = fully_connected_layer(fn1, shape=[8*8, 1], act_type='none', mean=meanDefault, sd=sdDefault, bias=biasDefault)
    return (input_layer, fn4)
