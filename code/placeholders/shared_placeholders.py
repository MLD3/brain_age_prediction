import tensorflow as tf
import numpy as np
from utils.config import get

def TrainingPlaceholder():
    """
    Returns a boolean placeholder that determines whether or not the model is training or testing.
    Used for things like dropout. Default value is False.
    """
    return tf.placeholder_with_default(False, shape=(), name="isTraining")

def MatrixPlaceholders():
    """
    Returns input and output placeholders for the connectivity matrices in the data file,
    """
    matricesPL = tf.placeholder(dtype=tf.float32, shape=(None, get('DATA.MATRICES.DIMENSION'), get('DATA.MATRICES.DIMENSION'), 1), name='matricesPL')
    labelsPL = tf.placeholder(dtype=tf.float32, shape=(None,1), name='labelsPL')
    return (matricesPL, labelsPL)

def AdamOptimizer(loss, learningRate):
    """
    Given the network loss, constructs the training op needed to train the
    network using adam optimization.

    Returns:
        the operation that begins the backpropogation through the network
        (i.e., the operation that minimizes the loss function).
    """
    optimizer = tf.train.AdamOptimizer(learningRate)
    trainOperation = optimizer.minimize(loss)
    return trainOperation

def ScheduledGradOptimizer(loss, baseLearningRate, momentum=0.9, decaySteps=1000, decayRate=0.5):
    """
    Given the network loss, constructs the training op needed to train the
    network using exponential learning rate decay and Nesterov Accelerated Gradient Descent with momentum.

    Returns:
        the operation that begins the backpropogation through the network
        (i.e., the operation that minimizes the loss function).
    """
    globalStep = tf.Variable(0, trainable=False, name='GlobalStep')
    learningRateTensor = tf.train.exponential_decay(baseLearningRate, globalStep, decaySteps, decayRate)
    optimizer = tf.train.MomentumOptimizer(learningRateTensor, momentum=momentum, use_nesterov=True)
    trainOperation = optimizer.minimize(loss, global_step=globalStep)
    return trainOperation