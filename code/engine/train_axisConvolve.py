import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetNPY import DataSetNPY
from model.build_sliceCNN import PartialConvolveCNN
from utils.saveModel import *
from utils.config import get
from engine.trainCommon import ModelTrainer
from placeholders.shared_placeholders import *

def ReshapeByAxis(inputLayer, batchAxis='X'):
    """
    Puts the indicated axis last, assuming that the input layer is
    shaped as [batch, X, Y, Z]
    """
    #Truncate the input to 121 pixels in each dimension
    #to ensure comparison among models
    inputLayer = inputLayer[:, 0:121, 0:121, 0:121]
    if batchAxis == 'X':
        return tf.transpose(inputLayer, perm=[0, 2, 3, 1])
    elif batchAxis == 'Y':
        return tf.transpose(inputLayer, perm=[0, 1, 3, 2])
    elif batchAxis == 'Z':
        return inputLayer

def GetTrainingOperation(trainLossOp, learningRate):
    with tf.variable_scope('optimizer'):
        trainUpdateOp = AdamOptimizer(trainLossOp, learningRate)
    return trainUpdateOp

def getMeanSquareError(inputBatch, labelBatch, trainingPL):
    kernelSizes = [(GlobalOpts.kernelSize, GlobalOpts.kernelSize)] * 3
    outputLayer = PartialConvolveCNN(trainInputBatch,
                                trainingPL,
                                kernelSizes=kernelSizes)
    return tf.losses.mean_square_error(labels=trainLabelBatch, predictions=trainOutputLayer)

def GetPartialConvolveCNN(
        trainDataSet,
        valdDataSet,
        testDataSet,
        trainingPL,
        batchAxis='X'):
    trainInputBatch, trainLabelBatch = trainDataSet.GetBatchOperations()
    trainInputBatch = ReshapeByAxis(trainInputBatch, batchAxis)
    trainLossOp = getMeanSquareError(trainInputBatch, trainLabelBatch, trainingPL)

    valdInputBatch, valdLabelBatch = valdDataSet.GetBatchOperations()
    valdInputBatch = ReshapeByAxis(valdInputBatch, batchAxis)
    valdLossOp = getMeanSquareError(valdInputBatch, valdLabelBatch, trainingPL)

    testInputBatch, testLabelBatch = testDataSet.GetBatchOperations()
    testInputBatch = ReshapeByAxis(testInputBatch, batchAxis)
    testLossOp = getMeanSquareError(testInputBatch, testLabelBatch, trainingPL)

    bootstrapInputBatch, bootstrapLabelBatch = testDataSet.GetRandomBatchOperations()
    bootstrapInputBatch = ReshapeByAxis(bootstrapInputBatch, batchAxis)
    bootstrapLossOp = getMeanSquareError(bootstrapInputBatch, bootstrapLabelBatch, trainingPL)

    return trainLossOp, valdLossOp, testLossOp, bootstrapLossOp

def GetDataSetInputs():
    with tf.variable_scope('Inputs'):
        with tf.variable_scope('TrainingInputs'):
            trainDataSet = DataSetNPY(filenames=GlobalOpts.trainFiles,
                                      imageBaseString=GlobalOpts.imageBaseString,
                                      imageBatchDims=GlobalOpts.imageBatchDims,
                                      batchSize=GlobalOpts.trainBatchSize)
        with tf.variable_scope('ValidationInputs'):
            valdDataSet  = DataSetNPY(filenames=GlobalOpts.valdFiles,
                                    imageBaseString=GlobalOpts.imageBaseString,
                                    imageBatchDims=GlobalOpts.imageBatchDims,
                                    batchSize=1,
                                    maxItemsInQueue=75,
                                    shuffle=False)
        with tf.variable_scope('TestInputs'):
            testDataSet  = DataSetNPY(filenames=GlobalOpts.testFiles,
                                    imageBaseString=GlobalOpts.imageBaseString,
                                    imageBatchDims=GlobalOpts.imageBatchDims,
                                    batchSize=1,
                                    maxItemsInQueue=75,
                                    shuffle=False)
    return trainDataSet, valdDataSet, testDataSet

def RunTestOnDirs(modelTrainer):
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    trainingPL = TrainingPlaceholder()
    learningRates = [0.001, 0.0001, 0.00001]
    names = []; trainUpdateOps = [];
    trainLossOp, valdLossOp, testLossOp, bootstrapLossOp = \
        GetPartialConvolveCNN(
            trainDataSet,
            valdDataSet,
            testDataSet,
            trainingPL,
            batchAxis=GlobalOpts.axis)
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
        modelTrainer.CompareRuns(sess, trainingPL, trainUpdateOps, trainLossOp, valdLossOp, testLossOp, names, bootstrapLossOp)

def defineDirs(dirSuffix=''):
    if GlobalOpts.axis == 'X':
        GlobalOpts.summaryDir = '{}xAxisChannels{}/'.format(get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'), dirSuffix)
        GlobalOpts.checkpointDir = '{}xAxisChannels{}/'.format(get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'), dirSuffix)
    elif GlobalOpts.axis == 'Y':
        GlobalOpts.summaryDir = '{}yAxisChannels{}/'.format(get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'), dirSuffix)
        GlobalOpts.checkpointDir = '{}yAxisChannels{}/'.format(get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'), dirSuffix)
    elif GlobalOpts.axis == 'Z':
        GlobalOpts.summaryDir = '{}zAxisChannels{}/'.format(get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'), dirSuffix)
        GlobalOpts.checkpointDir = '{}zAxisChannels{}/'.format(get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'), dirSuffix)
    else:
        raise ValueError('Unrecognized argument for parameter --axis: {}'.format(GlobalOpts.axis))

if __name__ == '__main__':
    additionalArgs = [{
            'flag': '--axis',
            'help': 'The axis to treat as channels (convolutions are preformed over the other two axes). One of X, Y, Z.',
            'action': 'store',
            'type': str,
            'dest': 'axis',
            'required': True
            }]
    ParseArgs('Run 2D CNN over different axes of MRI volumes', additionalArgs=additionalArgs)
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.NUMPY_PATH')
    GlobalOpts.imageBatchDims = (-1, 121, 145, 121)
    GlobalOpts.trainBatchSize = 8
    modelTrainer = ModelTrainer()

    GlobalOpts.kernelSize = 3
    defineDirs('{}x{}'.format(GlobalOpts.kernelSize, GlobalOpts.kernelSize))
    RunTestOnDirs(modelTrainer)
