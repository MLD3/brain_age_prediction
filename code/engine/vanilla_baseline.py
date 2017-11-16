import numpy as np
import pandas as pd
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, GridSearchCV, train_test_split
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

def ReportProgress(sess, step, lossFunction, imagesPL, labelsPL, splitTrainSet, splitTestSet):
    if step % 1 == 0:
        trainFeedDict = DefineFeedDict(splitTrainSet, imagesPL, labelsPL)
        trainingLoss = GetEvaluatedLoss(sess, splitTrainSet, lossFunction, imagesPL, labelsPL)
        testFeedDict = DefineFeedDict(splitTestSet, imagesPL, labelsPL)
        validationLoss = GetEvaluatedLoss(sess, splitTestSet, lossFunction, imagesPL, labelsPL)
        print('Step: %d, Evaluated Training Loss: %f, Evaluated Test Loss: %f' % (step, trainingLoss, validationLoss))


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
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
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


def test(dataSet):
    print('----------------------------------------------------------------')
    print('----------------------------TEST 1------------------------------')
    print('----------------------------------------------------------------')
    imagesPL, predictionLayer = fMRI_4D_CNN()
    labelsPL = tf.placeholder(tf.float32, shape=[None, 1])
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    trainOperation = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(lossFunction, global_step=global_step)

    RepeatModel(dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, numRepeats=10)

if __name__ == '__main__':
    dataHolder = DataHolder(readCSVData(get('DATA.SAMPLE.PATH')))
    dataHolder.getNIIImagesFromPath(get('DATA.IMAGES.PATH'))
    # print(len(dataHolder.matrices))
    dataSet = dataHolder.returnNIIDataset()

    test(dataSet)
