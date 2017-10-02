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
    if step % 100 == 0:
        trainFeedDict = DefineFeedDict(splitTrainSet, imagesPL, labelsPL)
        trainingLoss = GetEvaluatedLoss(sess, splitTrainSet, lossFunction, imagesPL, labelsPL)
        testFeedDict = DefineFeedDict(splitTestSet, imagesPL, labelsPL)
        validationLoss = GetEvaluatedLoss(sess, splitTestSet, lossFunction, imagesPL, labelsPL)
        print('Step: %d, Evaluated Training Loss: %f, Evaluated Test Loss: %f', (step, trainingLoss, validationLoss))

def TrainModel(sess, dataSet, imagesPL, labelsPL, predicitonLayer, trainOperation, lossFunction):
    X_train, X_test, y_train, y_test = train_test_split(trainSet.images, trainSet.labels, test_size=0.2)
    splitTrainSet = DataSet(X_train, y_train)
    splitTestSet = DataSet(X_test, y_test)

    for batch_index in range(get('TRAIN.CNN.NB_STEPS')):
        batch_images, batch_labels = splitTrainSet.next_batch(
            get('TRAIN.CNN.BATCH_SIZE'))
        feed_dict = DefineFeedDict(DataSet(batch_images, batch_labels), imagesPL, labelsPL)
        sess.run(trainingOperation, feed_dict=feed_dict)
        ReportProgress(sess, batch_index, lossFunction, imagesPL, labelsPL, splitTrainSet, splitTestSet)

if __name__ == '__main__':
    dataHolder = DataHolder(readCSVData(get('DATA.PHENOTYPICS.PATH')))
    dataHolder.getMatricesFromPath(get('DATA.MATRICES.PATH'))
    dataHolder.matricesToImages()
    dataSet = dataHolder.returnDataSet()

    imagesPL, predictionLayer = build_cnn()
    labelsPL = tf.placeholder(tf.float32, shape=[None, 1])
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)
    trainOperation = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(lossFunction)

    hooks  = [tf.train.StopAtStepHook(last_step=get('TRAIN.CNN.NB_STEPS'))]
    with tf.train.MonitoredSession(
            hook=hooks
    ) as sess:
        sess.run(tf.global_variables_initializer())
        TrainModel(sess, dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction)
