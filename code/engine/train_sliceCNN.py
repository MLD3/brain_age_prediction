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
    valdLossOp = tf.losses.mean_squared_error(labels=valdLabelBatch, predictions=valdOutputLayer)

    testInputBatch, testLabelBatch = testDataSet.GetBatchOperations()
    testOutputLayer = SliceCNN(testInputBatch,
                               trainingPL,
                               keepProbability=get('TRAIN.CNN_BASELINE.%s' % keepProbName),
                               optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                               downscaleRate=downscaleRate)
    testLossOp = tf.losses.mean_squared_error(labels=testLabelBatch, predictions=testOutputLayer)

    randomTestInput, randomTestLabels = testDataSet.GetRandomResamples()
    bootstrapOutputLayer = SliceCNN(randomTestInput,
                               trainingPL,
                               keepProbability=get('TRAIN.CNN_BASELINE.%s' % keepProbName),
                               optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                               downscaleRate=downscaleRate)
    bootstrapLossOp = tf.losses.mean_squared_error(labels=randomTestLabels, predictions=bootstrapOutputLayer)

    return trainUpdateOp, trainLossOp, valdLossOp, testLossOp, bootstrapLossOp

def RunTestOnDirs(modelTrainer, saveName, trainFiles, valdFiles, testFiles):
    tf.reset_default_graph()
    with tf.variable_scope('TrainingInputs'):
        trainDataSet = DataSetBIN(binFileNames=trainFiles)
    with tf.variable_scope('ValidationInputs'):
        valdDataSet  = DataSetBIN(binFileNames=valdFiles, batchSize=100, maxItemsInQueue=5000, shuffle=False)
    with tf.variable_scope('TestInputs'):
        testDataSet  = DataSetBIN(binFileNames=testFiles, batchSize=100, maxItemsInQueue=5000, shuffle=False)
    trainingPL = TrainingPlaceholder()
    trainUpdateOp, trainLossOp, valdLossOp, testLossOp, bootstrapLossOp = \
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
                                testDataSet)
    with tf.Session() as sess:
        modelTrainer.TrainModel(sess, trainingPL, trainUpdateOp, trainLossOp, valdLossOp, testLossOp, bootstrapLossOp)

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
        RunTestOnDirs(modelTrainer, 'xAxisSlices', [xTrainFile], [xValdFile], [xTestFile])
    elif args.data == 'Y':
        RunTestOnDirs(modelTrainer, 'xAxisSlices', [yTrainFile], [yValdFile], [yTestFile])
    elif args.data == 'Z':
        RunTestOnDirs(modelTrainer, 'xAxisSlices', [zTrainFile], [zValdFile], [zTestFile])
    elif args.data == 'XYZ':
        RunTestOnDirs(modelTrainer, 'xAxisSlices', [xTrainFile, yTrainFile, zTrainFile],
                                                   [xValdFile, yValdFile, zValdFile],
                                                   [xTestFile, yTestFile, zTestFile])
    elif args.data == 'ALL':
        RunTestOnDirs(modelTrainer, 'xAxisSlices', [xTrainFile], [xValdFile], [xTestFile])
        RunTestOnDirs(modelTrainer, 'xAxisSlices', [yTrainFile], [yValdFile], [yTestFile])
        RunTestOnDirs(modelTrainer, 'xAxisSlices', [zTrainFile], [zValdFile], [zTestFile])
        RunTestOnDirs(modelTrainer, 'xAxisSlices', [xTrainFile, yTrainFile, zTrainFile],
                                                   [xValdFile, yValdFile, zValdFile],
                                                   [xTestFile, yTestFile, zTestFile])
    else:
        print('Unrecognized data argument {}.'.format(args.data))
