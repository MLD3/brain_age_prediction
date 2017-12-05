import numpy as np
import pandas as pd
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, GridSearchCV, train_test_split, KFold
from sklearn.metrics import mean_squared_error
from utils.config import get, is_file_prefix
from model.fMRI_cnn import *
from data_scripts.DataSet import DataSet

def DefineFeedDict(dataSet, imagesPL, labelsPL):
    feed_dict = {
        imagesPL: dataSet.images,
        labelsPL: dataSet.labels
    }
    return feed_dict

def GetEvaluatedLoss(sess, dataSet, lossFunction, imagesPL, labelsPL):
    feed_dict = DefineFeedDict(dataSet, imagesPL, labelsPL)
    return sess.run(lossFunction, feed_dict=feed_dict)


def ReportProgress(sess, step, lossFunction, imagesPL, labelsPL, train_dataSet, test_dataSet):    
    if step % 10 == 0:
        # trainFeedDict = DefineFeedDict(train_dataSet, imagesPL, labelsPL)
        trainingLoss = GetEvaluatedLoss(sess, train_dataSet, lossFunction, imagesPL, labelsPL)
        # _, trainingLoss = sess.run([trainOperation, lossFunction])
        # print("Training Loss: ", trainingLoss)
        # testFeedDict = DefineFeedDict(test_dataSet, imagesPL, labelsPL)
        validationLoss = GetEvaluatedLoss(sess, test_dataSet, lossFunction, imagesPL, labelsPL)
        print('Step: %d, Evaluated Training Loss: %f, Evaluated Test Loss: %f' % (step, trainingLoss, validationLoss))

def TrainModel(sess, train_dataSet, test_dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, eval_function):
    # X_train = train_dataSet.images
    # y_train = train_dataSet.labels

    # X_test = test_dataSet.images
    # y_test = test_dataSet.labels

    for batch_index in range(get('TRAIN.VANILLA_BASELINE.NB_STEPS')):
        # batch_images, batch_labels = tf.train.shuffle_batch([X_train, X_test], batch_size = get('TRAIN.VANILLA_BASELINE.BATCH_SIZE'), capacity = get('TRAIN.VANILLA_BASELINE.CAPACITY'), min_after_dequeue = get('TRAIN.VANILLA_BASELINE.MIN_AFTER_DEQUEUE'))
        batch_images, batch_labels = train_dataSet.next_batch(
            get('TRAIN.VANILLA_BASELINE.BATCH_SIZE'))
        # print("Size from next_batch: ", batch_images.shape)
        feed_dict = DefineFeedDict(DataSet(batch_images, batch_labels), imagesPL, labelsPL)
        # tf.train.start_queue_runners(sess = sess)
        sess.run(trainOperation, feed_dict=feed_dict)
        # sess.run([batch_images, batch_labels])
        # sess.run(trainOperation)
        ReportProgress(sess, batch_index, lossFunction, imagesPL, labelsPL, DataSet(batch_images, batch_labels), test_dataSet)
        # ReportProgress(sess, batch_index, lossFunction, trainOperation)
    trainingLoss = GetEvaluatedLoss(sess, DataSet(batch_images, batch_labels), lossFunction, imagesPL, labelsPL)
    testLoss = GetEvaluatedLoss(sess, test_dataSet, lossFunction, imagesPL, labelsPL)
    eval_value = GetEvaluatedLoss(sess, test_dataSet, eval_function, imagesPL, labelsPL)
    print("After training, r-squared value is ", eval_value)
    return (trainingLoss, testLoss, eval_value)


def RepeatModel(train_dataSet, test_dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, eval_function, numRepeats=3):
    trainingLosses = []
    testLosses = []
    eval_values = []
    for i in range(numRepeats):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            (trainingLoss, testLoss, eval_value) = TrainModel(sess, train_dataSet, test_dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, eval_function)
            trainingLosses.append(trainingLoss)
            testLosses.append(testLoss)
            eval_values.append(eval_value)
    print("Mean Evaluated Training Loss: %f" % np.mean(trainingLosses))
    print("SD   Evaluated Training Loss: %f" % np.std(trainingLosses))
    print("Mean Evaluated Test Loss: %f" % np.mean(testLosses))
    print("SD   Evaluated Test Loss: %f" % np.std(testLosses))


def test(train_dataSet, test_dataSet):
    print('----------------------------------------------------------------')
    print('----------------------------TEST 1------------------------------')
    print('----------------------------------------------------------------')
    imagesPL, predictionLayer = fMRI_4D_CNN()
    labelsPL = tf.placeholder(tf.float32, shape=[None, 1])
    total_error = tf.reduce_sum(tf.square(tf.subtract(labelsPL, tf.reduce_mean(labelsPL))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(labelsPL, predictionLayer)))
    eval_function = tf.div(total_error, unexplained_error)
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    trainOperation = tf.train.AdamOptimizer(get('TRAIN.VANILLA_BASELINE.LEARNING_RATE')).minimize(lossFunction, global_step=global_step)

    RepeatModel(train_dataSet, test_dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, eval_function, numRepeats=3)
    # TrainModel(dataSet)


if __name__ == '__main__':
    dataHolder = DataHolder(readCSVData(get('DATA.SAMPLE.TRAIN_PATH')))
    subjects_id = dataHolder.getAllsubjects()
    subjects_id = np.asarray(subjects_id)
    kf = KFold(n_splits=5, shuffle=True)

    for train_index, test_index in kf.split(subjects_id):        
        print(len(subjects_id))
        print(test_index)
        train_ids, test_ids = subjects_id[train_index], subjects_id[test_index]
        # train_ids = subjects_id[train_index]
        # test_ids = subjects_id[test_index]
        dataHolder.getNIIImagesFromPath(get('DATA.IMAGES.TRAIN_PATH'), train_ids, test_ids)
        train_dataSet, test_dataSet = dataHolder.returnNIIDataset()
        print("Number of images for training data, 120 for each patient, ", len(dataHolder.train_images))
        test(train_dataSet, test_dataSet)
    # dataHolder.getNIIImagesFromPath(get('DATA.IMAGES.TRAIN_PATH'))
    
    # print(len(dataHolder.matrices))
    # train_dataSet, test_dataSet = dataHolder.returnNIIDataset()

    # test_dataHolder = DataHolder(readCSVData(get('DATA.SAMPLE.TEST_PATH')))
    # test_dataHolder.getNIIImagesFromPath(get('DATA.IMAGES.TEST_PATH'))
    # test_dataSet = test_dataHolder.returnNIIDataset()
    # # print(dataSet.images.shape)
    # # test(dataSet)
    # print(train_dataSet.images.shape)
    # print(test_dataSet.images.shape)

    # dataHolder = DataHolder(readCSVData(get('DATA.SAMPLE.TRAIN_PATH')))


    # test(train_dataSet, test_dataSet)
