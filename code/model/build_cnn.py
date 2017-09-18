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
    input_layer = tf.placeholder(tf.float32, shape=[None, 128*128*3])
    conv1 = conv_layer(input_layer, input_size=128, input_channels=3,
                    filter_size=16, output_channels=32, padding='SAME',
                    act_type='relu', mean=0.0, sd=0.01, bias=0.1, stride=4)
    pool1 = pool_layer(conv1, input_size=32, input_channels=32,
                        k_length=3, stride=2, padding='SAME')
    conv2 = conv_layer(pool1, input_size=16, input_channels=32,
                    filter_size=4, output_channels=64, padding='SAME',
                    act_type='relu', mean=0.0, sd=0.01, bias=1.0, stride=2)
    conv3 = conv_layer(conv2, input_size=8, input_channels=64,
                    filter_size=4, output_channels=128, padding='SAME',
                    act_type='relu', mean=0.0, sd=0.01, bias=1.0, stride=2)
    conv4 = conv_layer(conv3, input_size=4, input_channels=128,
                    filter_size=2, output_channels=128, padding='SAME',
                    act_type='relu', mean=0.0, sd=0.01, bias=1.0, stride=1)
    fc1 = fully_connected_layer(conv4, shape=[4 * 4 * 128, 128],
                    act_type='relu', mean=0.0, sd=0.01, bias=1.0)
    fc2 = fully_connected_layer(fc1, shape=[128, 2],
                    act_type='none', mean=0.0, sd=0.01, bias=1.0)

    return (input_layer, fc2)
