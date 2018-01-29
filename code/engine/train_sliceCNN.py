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
    trainOutputLayer = SliceCNN(trainInputBatch,
                                trainingPL,
                                keepProbability=get('TRAIN.CNN_BASELINE.%s' % keepProbName),
                                optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                                downscaleRate=downscaleRate)
    trainLossOp = tf.losses.mean_squared_error(labels=trainLabelBatch, predictions=trainOutputLayer)
    trainUpdateOp = AdamOptimizer(trainLossOp, get('TRAIN.CNN_BASELINE.%s' % learningRateName))

    valdInputBatch, valdLabelBatch = valdDataSet.GetBatchOperations()
    valdOutputLayer = SliceCNN(valdInputBatch,
                               trainingPL,
                               keepProbability=get('TRAIN.CNN_BASELINE.%s' % keepProbName),
                               optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                               downscaleRate=downscaleRate)
    valdLossOp = tf.losses.mean_squared_error(labels=valdLabelBatch,
                                              predictions=tf.reduce_mean(valdOutputLayer))

    testInputBatch, testLabelBatch = testDataSet.GetBatchOperations()
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
    with tf.Session() as sess:
        modelTrainer.TrainModel(sess, trainingPL, trainUpdateOp, trainLossOp, valdLossOp, testLossOp)

def WriteDefaultGraphToDir(dirName):
    writer = tf.summary.FileWriter(logdir=dirName, graph=tf.get_default_graph())
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run 2D Convolution Slice Tests on sMRI')
    parser.add_argument('--data', help='The data set to use. One of X, Y, Z, XYZ, ALL.', action='store', dest='data')
    args = parser.parse_args()

    xTrainFile = get('DATA.SLICES.X_SLICES_TRAIN')
    xValdFile  = get('DATA.SLICES.X_SLICES_VALD')
    xTestFile  = get('DATA.SLICES.X_SLICES_TEST')

    yTrainFile = get('DATA.SLICES.Y_SLICES_TRAIN')
    yValdFile  = get('DATA.SLICES.Y_SLICES_VALD')
    yTestFile  = get('DATA.SLICES.Y_SLICES_TEST')

    zTrainFile = get('DATA.SLICES.Z_SLICES_TRAIN')
    zValdFile  = get('DATA.SLICES.Z_SLICES_VALD')
    zTestFile  = get('DATA.SLICES.Z_SLICES_TEST')

    modelTrainer = ModelTrainerBIN()

    if args.data == 'X':
        with tf.variable_scope('xAxisModel'):
            RunTestOnDirs(modelTrainer, 'xAxisSlices', [xTrainFile], [xValdFile], [xTestFile], axis=0)
    elif args.data == 'Y':
        with tf.variable_scope('yAxisModel'):
            RunTestOnDirs(modelTrainer, 'yAxisSlices', [yTrainFile], [yValdFile], [yTestFile], axis=1)
    elif args.data == 'Z':
        with tf.variable_scope('zAxisModel'):
            RunTestOnDirs(modelTrainer, 'zAxisSlices', [zTrainFile], [zValdFile], [zTestFile], axis=2)
    elif args.data == 'XYZ':
        with tf.variable_scope('xyzAxisModel'):
            RunTestOnDirs(modelTrainer, 'xyzAxisSlices', [xTrainFile, yTrainFile, zTrainFile],
                                                   [xValdFile, yValdFile, zValdFile],
                                                   [xTestFile, yTestFile, zTestFile], axis=3)
    elif args.data == 'ALL':
        with tf.variable_scope('xAxisModel'):
            RunTestOnDirs(modelTrainer, 'xAxisSlices', [xTrainFile], [xValdFile], [xTestFile], axis=0)
        with tf.variable_scope('yAxisModel'):
            RunTestOnDirs(modelTrainer, 'yAxisSlices', [yTrainFile], [yValdFile], [yTestFile]. axis=1)
        with tf.variable_scope('zAxisModel'):
            RunTestOnDirs(modelTrainer, 'zAxisSlices', [zTrainFile], [zValdFile], [zTestFile], axis=2)
        with tf.variable_scope('xyzAxisModel'):
            RunTestOnDirs(modelTrainer, 'xyzAxisSlices', [xTrainFile, yTrainFile, zTrainFile],
                                                         [xValdFile, yValdFile, zValdFile],
                                                         [xTestFile, yTestFile, zTestFile], axis=3)
        WriteDefaultGraphToDir(dirName='{}{}'.format(get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'),
                                                     modelTrainer.dateString))
    else:
        print('Unrecognized data argument {}.'.format(args.data))
