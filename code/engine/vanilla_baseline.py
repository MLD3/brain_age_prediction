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

# def ReportProgress(sess, step, lossFunction, imagesPL, labelsPL, splitTrainSet, splitTestSet):
def ReportProgress(sess, step, lossFunction, trainOperation):    
    if step % 10 == 0:
        trainFeedDict = DefineFeedDict(splitTrainSet, imagesPL, labelsPL)
        trainingLoss = GetEvaluatedLoss(sess, splitTrainSet, lossFunction, imagesPL, labelsPL)
        # _, trainingLoss = sess.run([trainOperation, lossFunction])
        # print("Training Loss: ", trainingLoss)
        testFeedDict = DefineFeedDict(splitTestSet, imagesPL, labelsPL)
        validationLoss = GetEvaluatedLoss(sess, splitTestSet, lossFunction, imagesPL, labelsPL)
        print('Step: %d, Evaluated Training Loss: %f, Evaluated Test Loss: %f' % (step, trainingLoss, validationLoss))


def TrainModel(sess, dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction):
    X_train, X_test, y_train, y_test = train_test_split(dataSet.images, dataSet.labels, test_size=0.2)
    splitTrainSet = DataSet(X_train, y_train)
    splitTestSet = DataSet(X_test, y_test)

    for batch_index in range(get('TRAIN.VANILLA_BASELINE.NB_STEPS')):
        # batch_images, batch_labels = tf.train.shuffle_batch([X_train, X_test], batch_size = get('TRAIN.VANILLA_BASELINE.BATCH_SIZE'), capacity = get('TRAIN.VANILLA_BASELINE.CAPACITY'), min_after_dequeue = get('TRAIN.VANILLA_BASELINE.MIN_AFTER_DEQUEUE'))
        batch_images, batch_labels = splitTrainSet.next_batch(
            get('TRAIN.VANILLA_BASELINE.BATCH_SIZE'))
        feed_dict = DefineFeedDict(DataSet(batch_images, batch_labels), imagesPL, labelsPL)
        # tf.train.start_queue_runners(sess = sess)
        # sess.run(trainOperation, feed_dict=feed_dict)
        ReportProgress(sess, batch_index, lossFunction, imagesPL, labelsPL, splitTrainSet, splitTestSet)
        # ReportProgress(sess, batch_index, lossFunction, trainOperation)

    trainingLoss = GetEvaluatedLoss(sess, splitTrainSet, lossFunction, imagesPL, labelsPL)
    testLoss = GetEvaluatedLoss(sess, splitTestSet, lossFunction, imagesPL, labelsPL)
    return (trainingLoss, testLoss)
    # return trainingLoss


def RepeatModel(dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, numRepeats=10):
    trainingLosses = []
    testLosses = []
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    for i in range(numRepeats):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # trainingLoss = TrainModel(sess, dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction)
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
    # total_error = tf.reduce_sum(tf.square(tf.sub(labelsPL, tf.reduce_mean(labelsPL))))
    # unexplained_error = tf.reduce_sum(tf.square(tf.sub(labelsPL, predictionLayer)))
    # lossFunction = tf.sub(1, tf.div(total_error, unexplained_error))
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    trainOperation = tf.train.AdamOptimizer(get('TRAIN.VANILLA_BASELINE.LEARNING_RATE')).minimize(lossFunction, global_step=global_step)

    RepeatModel(dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, numRepeats=10)

if __name__ == '__main__':
    dataHolder = DataHolder(readCSVData(get('DATA.SAMPLE.PATH')))
    dataHolder.getNIIImagesFromPath(get('DATA.IMAGES.PATH'))
    # print(len(dataHolder.matrices))
    dataSet = dataHolder.returnNIIDataset()
    # print(dataSet.images.shape)
    test(dataSet)
