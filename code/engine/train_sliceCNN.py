import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetBIN import DataSetBIN
from model.build_sliceCNN import SliceCNN
from utils.saveModel import *
from utils.config import get
from engine.trainCommon import ModelTrainer
from placeholders.shared_placeholders import *

def GetTrainingOperation(trainLossOp, learningRate):
    with tf.variable_scope('optimizer'):
        trainUpdateOp = AdamOptimizer(trainLossOp, learningRate)
    return trainUpdateOp

def GetSliceCNN(
        trainDataSet,
        valdDataSet,
        testDataSet,
        trainingPL):
    trainInputBatch, trainLabelBatch = trainDataSet.GetBatchOperations()
    trainOutputLayer = SliceCNN(trainInputBatch,
                                trainingPL)
    trainLossOp = tf.losses.mean_squared_error(labels=trainLabelBatch, predictions=trainOutputLayer)

    valdInputBatch, valdLabelBatch = valdDataSet.GetBatchOperations()
    valdOutputLayer = SliceCNN(valdInputBatch,
                               trainingPL)
    valdLossOp = tf.losses.mean_squared_error(labels=valdLabelBatch,
                                              predictions=tf.reduce_mean(valdOutputLayer))

    testInputBatch, testLabelBatch = testDataSet.GetBatchOperations()
    testOutputLayer = SliceCNN(testInputBatch,
                               trainingPL)
    testLossOp = tf.losses.mean_squared_error(labels=testLabelBatch,
                                              predictions=tf.reduce_mean(testOutputLayer))

    return trainLossOp, valdLossOp, testLossOp

def GetDataSetInputs():
    with tf.variable_scope('Inputs'):
        with tf.variable_scope('TrainingInputs'):
            trainDataSet = DataSetBIN(binFileNames=GlobalOpts.trainFiles,
                                      imageDims=GlobalOpts.trainImageDims,
                                      batchSize=GlobalOpts.trainBatchSize)
        with tf.variable_scope('ValidationInputs'):
            valdDataSet  = DataSetBIN(binFileNames=GlobalOpts.valdFiles,
                                    imageDims=GlobalOpts.testImageDims,
                                    batchSize=1,
                                    maxItemsInQueue=75,
                                    minItemsInQueue=1,
                                    shuffle=False,
                                    spliceInputAlongAxis=GlobalOpts.axis)
        with tf.variable_scope('TestInputs'):
            testDataSet  = DataSetBIN(binFileNames=GlobalOpts.testFiles,
                                    imageDims=GlobalOpts.testImageDims,
                                    batchSize=1,
                                    maxItemsInQueue=75,
                                    minItemsInQueue=1,
                                    shuffle=False,
                                    spliceInputAlongAxis=GlobalOpts.axis)
    return trainDataSet, valdDataSet, testDataSet

def RunTestOnDirs(modelTrainer):
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    trainingPL = TrainingPlaceholder()
    learningRates = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    names = []; trainUpdateOps = [];
    trainLossOp, valdLossOp, testLossOp = \
        GetSliceCNN(
            trainDataSet,
            valdDataSet,
            testDataSet,
            trainingPL)
    for rate in learningRates:
        name = 'learningRate_{}'.format(rate)
        with tf.variable_scope(name):
            trainUpdateOp = GetTrainingOperation(trainLossOp, rate)
            trainUpdateOps.append(trainUpdateOp)
            names.append(name)

    modelTrainer.DefineNewParams(GlobalOpts.summaryDir,
                                GlobalOpts.checkpointDir,
                                GlobalOpts.numSteps)
    config  = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GlobalOpts.gpuMemory
    with tf.Session(config=config) as sess:
        modelTrainer.CompareRuns(sess, trainingPL, trainUpdateOps, trainLossOp, valdLossOp, testLossOp, names)

if __name__ == '__main__':
    additionalArgs = [{
            'flag': '--data',
            'help': 'The data set to use. One of X, Y, Z, XYZ, 3D.',
            'action': 'store',
            'type': str,
            'dest': 'data',
            'required': True
            }]

    ParseArgs('Run 2D convolution tests over structural MRI slices', additionalArgs=additionalArgs)
    GlobalOpts.valdFiles = [get('DATA.BIN.VALD')]
    GlobalOpts.testFiles = [get('DATA.BIN.TEST')]
    GlobalOpts.trainImageDims = [145, 145, 1]
    GlobalOpts.testImageDims = [121, 145, 121, 1]
    GlobalOpts.trainBatchSize = 64

    modelTrainer = ModelTrainer()

    if GlobalOpts.data == 'X':
        GlobalOpts.summaryDir = get('TRAIN.CNN_BASELINE.SUMMARIES_DIR') + 'xAxisSlices/'
        GlobalOpts.checkpointDir = get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR') + 'xAxisSlices/'
        GlobalOpts.trainFiles = [get('DATA.BIN.X_SLICES_TRAIN')]
        GlobalOpts.axis = 0
    elif GlobalOpts.data == 'Y':
        GlobalOpts.summaryDir = get('TRAIN.CNN_BASELINE.SUMMARIES_DIR') + 'yAxisSlices/'
        GlobalOpts.checkpointDir = get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR') + 'yAxisSlices/'
        GlobalOpts.trainFiles = [get('DATA.BIN.Y_SLICES_TRAIN')]
        GlobalOpts.axis = 1
    elif GlobalOpts.data == 'Z':
        GlobalOpts.summaryDir = get('TRAIN.CNN_BASELINE.SUMMARIES_DIR') + 'zAxisSlices/'
        GlobalOpts.checkpointDir = get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR') + 'zAxisSlices/'
        GlobalOpts.trainFiles = [get('DATA.BIN.Z_SLICES_TRAIN')]
        GlobalOpts.axis = 2
    elif GlobalOpts.data == 'XYZ':
        GlobalOpts.summaryDir = get('TRAIN.CNN_BASELINE.SUMMARIES_DIR') + 'allAxesSlices/'
        GlobalOpts.checkpointDir = get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR') + 'allAxesSlices/'
        GlobalOpts.trainFiles = [get('DATA.BIN.X_SLICES_TRAIN'),
                                 get('DATA.BIN.Y_SLICES_TRAIN'),
                                 get('DATA.BIN.Z_SLICES_TRAIN')]
        GlobalOpts.axis = 3
    else:
        raise ValueError('Unrecognized argument for parameter --data: {}.'.format(args.data))

    RunTestOnDirs(modelTrainer)
