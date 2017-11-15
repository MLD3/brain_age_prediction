import os
import tensorflow as tf

def restore(sess, checkpoint_path):
    """
    If a checkpoint exists, restores the tensorflow model from the checkpoint.
    Returns the tensorflow Saver and the checkpoint filename.
    """
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    if checkpoint:
        path = checkpoint.model_checkpoint_path
        print('Restoring model parameters from {}'.format(path))
        saver.restore(sess, path)
    else:
        print('No saved model parameters found. Training from scratch...')
    # Return checkpoint path for call to saver.save()
    save_path = os.path.join(
        checkpoint_path, os.path.basename(os.path.dirname(checkpoint_path)))
    return saver, save_path
