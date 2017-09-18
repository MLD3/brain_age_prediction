import tensorflow as tf
from utils.config import get, is_file_prefix
from model.build_cnn import cnn

from preprocessing.labeling import getAugmentedDataSet, getDataSet

def get_weights(saver, sess):
    ''' load model weights if they were saved previously '''
    if is_file_prefix('TRAIN.CNN.CHECKPOINT'):
        saver.restore(sess, get('TRAIN.CNN.CHECKPOINT'))
        print('Yay! I restored weights from a saved model!')
    else:
        print('OK, I did not find a saved model, so I will start training from scratch!')

def save_model(sess, path):
    saver = tf.train.Saver()
    save_path = saver.save(sess, path)
    print("Model saved to file: " + str(save_path))

def report_training_progress(sess, batch_index, input_layer, loss_func, validationSet, accuracy):
    ''' Update user on training progress '''
    if batch_index % 5:
        return
    print('starting batch number %d \033[100D\033[1A' % batch_index)
    if batch_index % 50:
        return
    error = loss_func.eval(feed_dict={input_layer: validationSet.images, true_labels: validationSet.labels})
    acc = accuracy.eval(feed_dict={input_layer: validationSet.images, true_labels: validationSet.labels})
    print('\n \t cross_entropy is about %f' % error)
    print(' \t accuracy is about %f' % acc)
    if batch_index % 500:
        return
    print("Saving model...")
    save_model(sess, get('TRAIN.CNN.CHECKPOINT'))


def train_cnn(sess, input_layer, prediction_layer, loss_func, optimizer, trainingSet, validationSet, accuracy):
    ''' Train CNN '''
    try:
        for batch_index in range(get('TRAIN.CNN.NB_STEPS')):
            report_training_progress(sess,
                batch_index, input_layer, loss_func, validationSet, accuracy)
            batch_images, batch_labels = trainingSet.next_batch(
                get('TRAIN.CNN.BATCH_SIZE'))
            optimizer.run(
                feed_dict={input_layer: batch_images, true_labels: batch_labels})
    except KeyboardInterrupt:
        print('OK, I will stop training even though I am not finished.')

if __name__ == '__main__':
    trainingSet = getAugmentedDataSet(labelsFile='../data/train/labels.txt', imageDir='../data/train/cropped', imageExtension='.png', oneHot=True)
    validationSet = getDataSet(labelsFile='../data/validation/labels.txt', imageDir='../data/validation/cropped', imageExtension='.png', oneHot=True)
    testSet = getDataSet(labelsFile='../data/test/labels.txt', imageDir='../data/test/cropped', imageExtension='.png', oneHot=True)

    input_layer, prediction_layer = cnn()
    true_labels = tf.placeholder(tf.float32, shape=[None, 2])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=true_labels, logits=prediction_layer))
    prediction = tf.argmax(prediction_layer, axis=1)
    accuracy =  tf.contrib.metrics.accuracy(tf.argmax(true_labels, axis=1), prediction)
    optimizer = tf.train.AdamOptimizer(
                    get('TRAIN.CNN.LEARNING_RATE')).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    get_weights(saver, sess)

    print('training...')
    train_cnn(sess, input_layer, prediction_layer, cross_entropy, optimizer, trainingSet, validationSet, accuracy)

    validation_accuracy = accuracy.eval(feed_dict=
            {input_layer: validationSet.images,
             true_labels: validationSet.labels})
    print("My accuracy was: " + str(validation_accuracy))
