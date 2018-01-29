import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from data_scripts.DataSetBIN import DataSetBIN
from sklearn.model_selection import train_test_split, KFold
from model.build_sliceCNN import SliceCNN
from utils import saveModel
from utils.config import get
from engine.trainCommonBIN import ModelTrainerBIN
from placeholders.shared_placeholders import *

def GetSliceCNN(
        trainDataSet,
        valdDataSet,
        testDataSet,
        trainingPL,
        learningRateName='LEARNING_RATE',
        keepProbName='KEEP_PROB',
        optionalHiddenLayerUnits=0,
        downscaleRate=None):
    trainInputBatch, trainLabelBatch = trainDataSet.GetBatchOperations()
    with tf.variable_scope('trainCNN'):
        trainOutputLayer = SliceCNN(trainInputBatch,
                                    trainingPL,
                                    keepProbability=get('TRAIN.CNN_BASELINE.%s' % keepProbName),
                                    optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                                    downscaleRate=downscaleRate)
    trainLossOp = tf.losses.mean_squared_error(labels=trainLabelBatch, predictions=trainOutputLayer)
    with tf.variable_scope('Optimzer'):
        trainUpdateOp = AdamOptimizer(trainLossOp, get('TRAIN.CNN_BASELINE.%s' % learningRateName))

    valdInputBatch, valdLabelBatch = valdDataSet.GetBatchOperations()
    with tf.variable_scope('valdCNN'):
        valdOutputLayer = SliceCNN(valdInputBatch,
                                   trainingPL,
                                   keepProbability=get('TRAIN.CNN_BASELINE.%s' % keepProbName),
                                   optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                                   downscaleRate=downscaleRate)
    valdLossOp = tf.losses.mean_squared_error(labels=valdLabelBatch,
                                              predictions=tf.reduce_mean(valdOutputLayer))

    testInputBatch, testLabelBatch = testDataSet.GetBatchOperations()
    with tf.variable_scope('testCNN'):
        testOutputLayer = SliceCNN(testInputBatch,
                                   trainingPL,
                                   keepProbability=get('TRAIN.CNN_BASELINE.%s' % keepProbName),
                                   optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                                   downscaleRate=downscaleRate)
    testLossOp = tf.losses.mean_squared_error(labels=testLabelBatch,
                                              predictions=tf.reduce_mean(testOutputLayer))

    return trainUpdateOp, trainLossOp, valdLossOp, testLossOp

def RunTestOnDirs(modelTrainer, saveName, trainFiles, valdFiles, testFiles, axis=0):
    with tf.variable_scope('TrainingInputs'):
        trainDataSet = DataSetBIN(binFileNames=trainFiles)
    with tf.variable_scope('ValidationInputs'):
        valdDataSet  = DataSetBIN(binFileNames=valdFiles,
                                imageDims=[121, 145, 121, 1],
                                batchSize=1,
                                maxItemsInQueue=75,
                                minItemsInQueue=1,
                                shuffle=False,
                                training=False,
                                axis=axis)
    with tf.variable_scope('TestInputs'):
        testDataSet  = DataSetBIN(binFileNames=testFiles,
                                imageDims=[121, 145, 121, 1],
                                batchSize=1,
                                maxItemsInQueue=75,
                                minItemsInQueue=1,
                                shuffle=False,
                                training=False,
                                axis=axis)
    trainingPL = TrainingPlaceholder()
    trainUpdateOp, trainLossOp, valdLossOp, testLossOp = \
        GetSliceCNN(
            trainDataSet,
            valdDataSet,
            testDataSet,
            trainingPL,
            learningRateName='LEARNING_RATE',
            keepProbName='KEEP_PROB',
            optionalHiddenLayerUnits=0,
            downscaleRate=None)
    modelTrainer.DefineNewParams(saveName,
                                trainDataSet,
                                valdDataSet,
                                testDataSet,
                                numberOfSteps=get('TRAIN.DEFAULTS.TEST_NB_STEPS'))
    # with tf.Session() as sess:
        # modelTrainer.TrainModel(sess, trainingPL, trainUpdateOp, trainLossOp, valdLossOp, testLossOp)

def WriteDefaultGraphToDir(dirName):
    writer = tf.summary.FileWriter(logdir=dirName, graph=tf.get_default_graph())
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run 2D Convolution Slice Tests on sMRI')
    parser.add_argument('--data', help='The data set to use. One of X, Y, Z, XYZ, ALL.', action='store', dest='data')
    args = parser.parse_args()

    xTrainFile = get('DATA.SLICES.X_SLICES_TRAIN')
    yTrainFile = get('DATA.SLICES.Y_SLICES_TRAIN')
    zTrainFile = get('DATA.SLICES.Z_SLICES_TRAIN')

    valdFile = get('DATA.SLICES.VALD')
    testFile = get('DATA.SLICES.TEST')

    modelTrainer = ModelTrainerBIN()

    if args.data == 'X':
        with tf.variable_scope('xAxisModel'):
            RunTestOnDirs(modelTrainer, 'xAxisSlices', [xTrainFile], [valdFile], [testFile], axis=0)
    elif args.data == 'Y':
        with tf.variable_scope('yAxisModel'):
            RunTestOnDirs(modelTrainer, 'yAxisSlices', [yTrainFile], [valdFile], [testFile], axis=1)
    elif args.data == 'Z':
        with tf.variable_scope('zAxisModel'):
            RunTestOnDirs(modelTrainer, 'zAxisSlices', [zTrainFile], [valdFile], [testFile], axis=2)
    elif args.data == 'XYZ':
        with tf.variable_scope('xyzAxisModel'):
            RunTestOnDirs(modelTrainer, 'xyzAxisSlices', [xTrainFile, yTrainFile, zTrainFile],
                                                   [valdFile],
                                                   [testFile], axis=3)
    elif args.data == 'ALL':
        with tf.variable_scope('xAxisModel'):
            RunTestOnDirs(modelTrainer, 'xAxisSlices', [xTrainFile], [valdFile], [testFile], axis=0)
        with tf.variable_scope('yAxisModel'):
            RunTestOnDirs(modelTrainer, 'yAxisSlices', [yTrainFile], [valdFile], [testFile], axis=1)
        with tf.variable_scope('zAxisModel'):
            RunTestOnDirs(modelTrainer, 'zAxisSlices', [zTrainFile], [valdFile], [testFile], axis=2)
        with tf.variable_scope('xyzAxisModel'):
            RunTestOnDirs(modelTrainer, 'xyzAxisSlices', [xTrainFile, yTrainFile, zTrainFile],
                                                         [valdFile],
                                                         [testFile], axis=3)
        WriteDefaultGraphToDir(dirName='{}{}'.format(get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'),
                                                     modelTrainer.dateString))
        for x in tf.global_variables():
            print(x.name)

    else:
        print('Unrecognized data argument {}.'.format(args.data))
