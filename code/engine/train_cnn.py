import tensorflow as tf
from utils.config import get, is_file_prefix
from model.build_cnn import cnn

def get_weights(saver, sess):
    if is_file_prefix('TRAIN.CNN.CHECKPOINT'):
        saver.restore(sess, get('TRAIN.CNN.CHECKPOINT'))
        print('Restored weights from a checkpoint.')
    else:
        print('Training from scratch...')

def save_model(sess, path):
    saver = tf.train.Saver()
    save_path = saver.save(sess, path)
    print("Model saved to file: " + str(save_path))

def report_training_progress(sess, batch_index, input_layer, loss_func, validationSet):
    if batch_index % 5:
        return
    print('starting batch number %d \033[100D\033[1A' % batch_index)
    if batch_index % 50:
        return
    error = loss_func.eval(feed_dict={input_layer: validationSet.images, true_labels: validationSet.labels})
    print('\n \t Evaluated Loss Function: %f' % error)
    if batch_index % 500:
        return
    print("Saving model...")
    save_model(sess, get('TRAIN.CNN.CHECKPOINT'))


def train_cnn(sess, input_layer, prediction_layer, loss_func, optimizer, trainingSet, validationSet, accuracy):
    try:
        for batch_index in range(get('TRAIN.CNN.NB_STEPS')):
            report_training_progress(sess,
                batch_index, input_layer, loss_func, validationSet, accuracy)
            batch_images, batch_labels = trainingSet.next_batch(
                get('TRAIN.CNN.BATCH_SIZE'))
            optimizer.run(
                feed_dict={input_layer: batch_images, true_labels: batch_labels})
    except KeyboardInterrupt:
        print('Terminating training session due to keyboard exception...')

if __name__ == '__main__':
    ##########################################################################
    ############################# GET DATA SETS ##############################
    ##########################################################################

    ##########################################################################
    ################## Fetch CNN, define loss and optimzer ###################
    ##########################################################################


    ##########################################################################
    ################ Start an interactive Tensorflow Session #################
    ##########################################################################

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    get_weights(saver, sess)

    ##########################################################################
    ##################### Train the defined model.... ########################
    ##########################################################################
