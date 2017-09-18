import tensorflow as tf
from utils.config import get, is_file_prefix
from model.build_cnn import cnn
from sklearn.model_selection import StratifiedKFold

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


def trainCNN(sess, input_layer, prediction_layer, loss_func, optimizer, trainingSet, validationSet):
    try:
        for batch_index in range(get('TRAIN.CNN.NB_STEPS')):
            report_training_progress(sess,
                batch_index, input_layer, loss_func, validationSet)
            batch_images, batch_labels = trainingSet.next_batch(
                get('TRAIN.CNN.BATCH_SIZE'))
            optimizer.run(
                feed_dict={input_layer: batch_images, true_labels: batch_labels})
    except KeyboardInterrupt:
        print('Terminating training session due to keyboard exception...')

def fetchArchitecture(params):
    input_layer = #TODO
    prediction_layer = #TODO
    loss_func = #TODO
    optimzer = #TODO
    return (input_layer, prediction_layer, loss_func, optimzer)

def crossValidate(sess, trainingSet, k=5, params):
    validationPerformance = []
    skf = StratifiedKFold(n_splits=k)
    bestParam = #TODO
    for trainIndex, valdIndex in skf.split(trainingSet.images, trainingSet.labels)
        X_train, X_vald = trainingSet.images[trainIndex], trainingSet.images[valdIndex]
        y_train, y_vald = trainingSet.labels[trainIndex], trainingSet.labels[valdIndex]
        splitTrainSet = Dataset(X_train, y_train)
        validationSet = Dataset(X_vald, y_vald)
        (input_layer, prediction_layer, loss_func, optimzer) = fetchArchitecture(params)
        #TODO: Add some hyper parameter adjustment here
        trainCNN(sess, input_layer, prediction_layer, loss_func, optimizer, splitTrainSet, validationSet)
        perf = loss_func.eval(feed_dict={input_layer: X_vald, true_labels: y_vald})
        validationPerformance.append(perf)

    return (np.mean(validationPerformance), bestParam)

def performanceOnParameter(trainSet, sess, params):
    numCrossValidationTimes = 100
    for trial in range(numCrossValidationTimes):
        splitTrainSet = #TODO
        splitTestSet = #TODO
        valPerf, bestParam = crossValidate(sess, splitTrainSet, k=5, params)
        (input_layer, prediction_layer, loss_func, optimzer) = fetchArchitecture(params)
        trainCNN(sess, input_layer, prediction_layer, loss_func, optimizer, splitTrainSet, splitTestSet)
        error = loss_func.eval(feed_dict={input_layer: splitTestSet.images, true_labels: splitTestSet.labels})
        print("Performance was: " + '%f' % error)

if __name__ == '__main__':
    ##########################################################################
    ############################# GET DATA SETS ##############################
    ##########################################################################
    trainSet = []
    testSet = []

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
