import tensorflow as tf

def restore(sess, savePath):
    """
    If a checkpoint exists, restores the tensorflow model from the checkpoint.
    Returns the tensorflow Saver.
    """
    saver = tf.train.Saver()
    if tf.train.checkpoint_exists(savePath):
        try:
            saver.restore(sess, savePath)
            print('Restored model from {} successfully'.format(savePath))
        except Exception as error:
            print('Unable to restore model from path {} with error {}'.format(savePath, error))
    else:
        print('No checkpoint exists at path {}. Training from scratch...'.format(savePath))
    return saver

def WriteDefaultGraphToDir(dirName):
    writer = tf.summary.FileWriter(logdir=dirName, graph=tf.get_default_graph())
    writer.close()
