import os
import tensorflow as tf

def restore(sess, savePath):
    """
    If a checkpoint exists, restores the tensorflow model from the checkpoint.
    Returns the tensorflow Saver.
    """
    saver = tf.train.Saver()
    if os.path.exists(savePath):
        print('Restoring model parameters from {}'.format(savePath))
        saver.restore(sess, savePath)
    else:
        print('No saved model parameters found. Training from scratch...')
    return saver
