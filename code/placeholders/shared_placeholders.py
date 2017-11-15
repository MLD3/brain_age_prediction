import tensorflow as tf
import numpy as np
from utils import get

def trainingPlaceholder():
    """
    Returns a boolean placeholder that determines whether or not the model is training or testing.
    Used for things like dropout. Default value is False.
    """
    return tf.placeholder_with_default(False, shape=(), name="isTraining")

def matrixPlaceholders():
    """
    Returns input and output placeholders for the connectivity matrices in the data file,
    """
    matricesPL = tf.placeholder(dtype=tf.float32, shape=(None, get('DATA.MATRICES.DIMENSION'), get('DATA.MATRICES.DIMENSION'), 1), name='matricesPL')
    labelsPL = tf.placeholder(dtype=tf.float32, shape=(None,), name='labelsPL')
    return (matricesPL, labelsPL)
