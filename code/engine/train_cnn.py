import tensorflow as tf
import numpy as np
import pandas as pd
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from utils.config import get, is_file_prefix
from data_scripts.DataSet import DataSet
from model.build_cnn import *

def DefineFeedDict(dataSet, imagesPL, labelsPL):
    feed_dict = {
        imagesPL: dataSet.images,
        labelsPL: dataSet.labels
    }
    return feed_dict

def GetEvaluatedLoss(sess, dataSet, lossFunction, imagesPL, labelsPL):
    feed_dict = DefineFeedDict(dataSet, imagesPL, labelsPL)
    return sess.run(lossFunction, feed_dict=feed_dict)

def ReportProgress(sess, step, lossFunction, imagesPL, labelsPL, splitTrainSet, splitTestSet):
    if step % 10 == 0:
        trainFeedDict = DefineFeedDict(splitTrainSet, imagesPL, labelsPL)
        trainingLoss = GetEvaluatedLoss(sess, splitTrainSet, lossFunction, imagesPL, labelsPL)
        testFeedDict = DefineFeedDict(splitTestSet, imagesPL, labelsPL)
        validationLoss = GetEvaluatedLoss(sess, splitTestSet, lossFunction, imagesPL, labelsPL)
        print('Step: %d, Evaluated Training Loss: %f, Evaluated Test Loss: %f' % (step, trainingLoss, validationLoss))

def TrainModelXY(sess, X, Y, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    splitTrainSet = DataSet(X_train, y_train)
    splitTestSet = DataSet(X_test, y_test)

    for batch_index in range(get('TRAIN.CNN.NB_STEPS')):
        batch_images, batch_labels = splitTrainSet.next_batch(
            get('TRAIN.CNN.BATCH_SIZE'))
        feed_dict = DefineFeedDict(DataSet(batch_images, batch_labels), imagesPL, labelsPL)
        sess.run(trainOperation, feed_dict=feed_dict)
        ReportProgress(sess, batch_index, lossFunction, imagesPL, labelsPL, splitTrainSet, splitTestSet)

    trainingLoss = GetEvaluatedLoss(sess, splitTrainSet, lossFunction, imagesPL, labelsPL)
    testLoss = GetEvaluatedLoss(sess, splitTestSet, lossFunction, imagesPL, labelsPL)
    return (trainingLoss, testLoss)

def RepeatModelXY(X, Y, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, numRepeats=10):
    trainingLosses = []
    testLosses = []
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    for i in range(numRepeats):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            (trainingLoss, testLoss) = TrainModelXY(sess, X, Y, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction)
            trainingLosses.append(trainingLoss)
            testLosses.append(testLoss)
    print("Mean Evaluated Training Loss: %f" % np.mean(trainingLosses))
    print("SD   Evaluated Training Loss: %f" % np.std(trainingLosses))
    print("Mean Evaluated Test Loss: %f" % np.mean(testLosses))
    print("SD   Evaluated Test Loss: %f" % np.std(testLosses))

def TrainModel(sess, dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction):
    X_train, X_test, y_train, y_test = train_test_split(dataSet.images, dataSet.labels, test_size=0.2)
    splitTrainSet = DataSet(X_train, y_train)
    splitTestSet = DataSet(X_test, y_test)

    for batch_index in range(get('TRAIN.CNN.NB_STEPS')):
        batch_images, batch_labels = splitTrainSet.next_batch(
            get('TRAIN.CNN.BATCH_SIZE'))
        feed_dict = DefineFeedDict(DataSet(batch_images, batch_labels), imagesPL, labelsPL)
        sess.run(trainOperation, feed_dict=feed_dict)
        ReportProgress(sess, batch_index, lossFunction, imagesPL, labelsPL, splitTrainSet, splitTestSet)

    trainingLoss = GetEvaluatedLoss(sess, splitTrainSet, lossFunction, imagesPL, labelsPL)
    testLoss = GetEvaluatedLoss(sess, splitTestSet, lossFunction, imagesPL, labelsPL)
    return (trainingLoss, testLoss)

def RepeatModel(dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, numRepeats=10):
    trainingLosses = []
    testLosses = []
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    for i in range(numRepeats):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            (trainingLoss, testLoss) = TrainModel(sess, dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction)
            trainingLosses.append(trainingLoss)
            testLosses.append(testLoss)
    print("Mean Evaluated Training Loss: %f" % np.mean(trainingLosses))
    print("SD   Evaluated Training Loss: %f" % np.std(trainingLosses))
    print("Mean Evaluated Test Loss: %f" % np.mean(testLosses))
    print("SD   Evaluated Test Loss: %f" % np.std(testLosses))

def test1(dataSet):
    print('----------------------------------------------------------------')
    print('----------------------------TEST 1------------------------------')
    print('----------------------------------------------------------------')
    imagesPL, predictionLayer = cnnDefault()
    labelsPL = tf.placeholder(tf.float32, shape=[None, 1])
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    trainOperation = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(lossFunction, global_step=global_step)

    RepeatModel(dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, numRepeats=10)

def test2(dataSet):
    print('----------------------------------------------------------------')
    print('----------------------------TEST 2------------------------------')
    print('----------------------------------------------------------------')
    X = np.array([mat[np.tril_indices(mat.shape[0], k=-1)] for mat in dataSet.matrices])
    Y = np.array(dataSet._df['AgeYears'].values.copy())
    Y = np.reshape(Y, [Y.shape[0], 1])

    imagesPL, predictionLayer = cnnNeural()
    labelsPL = tf.placeholder(tf.float32, shape=[None, 1])
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    trainOperation = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(lossFunction, global_step=global_step)

    RepeatModelXY(X, Y, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, numRepeats=10)

def test3(dataSet):
    print('----------------------------------------------------------------')
    print('----------------------------TEST 3------------------------------')
    print('----------------------------------------------------------------')
    imagesPL, predictionLayer = cnnSmall()
    labelsPL = tf.placeholder(tf.float32, shape=[None, 1])
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    trainOperation = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(lossFunction, global_step=global_step)

    RepeatModel(dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, numRepeats=10)

def test4(dataSet):
    print('----------------------------------------------------------------')
    print('----------------------------TEST 4------------------------------')
    print('----------------------------------------------------------------')
    imagesPL, predictionLayer = cnn_larger_klength()
    labelsPL = tf.placeholder(tf.float32, shape=[None, 1])
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    trainOperation = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(lossFunction, global_step=global_step)

    RepeatModel(dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, numRepeats=10)

if __name__ == '__main__':
    dataHolder = DataHolder(readCSVData(get('DATA.PHENOTYPICS.PATH')))
    dataHolder.getMatricesFromPath(get('DATA.MATRICES.PATH'))
    dataHolder.matricesToImages()
    dataSet = dataHolder.returnDataSet()
    # test1(dataSet)
    test2(dataHolder)
    test3(dataSet)
    test4(dataSet)
