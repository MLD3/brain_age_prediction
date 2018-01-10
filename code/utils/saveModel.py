import tensorflow as tf

def restore(sess, savePath):
    """
    If a checkpoint exists, restores the tensorflow model from the checkpoint.
    Returns the tensorflow Saver.
    """
    saver = tf.train.Saver()
    try:
        saver.restore(sess, savePath)
        print('Restored model from {} successfully'.format(savePath))
    except ValueError as error:
        print('Unable to restore model from path {} with error {}'.format(savePath, error))

    return saver
