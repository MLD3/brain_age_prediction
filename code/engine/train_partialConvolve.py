import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetNPY import DataSetNPY
from model.build_baselineStructuralCNN import depthPatchCNN, batchPatchCNN, PatchCNN2Batch1Depth
from utils.saveModel import *
from utils.config import get
from engine.trainCommon import ModelTrainer
from placeholders.shared_placeholders import *

def GetTrainingOperation(trainLossOp, learningRate):
    with tf.variable_scope('optimizer'):
        trainUpdateOp = AdamOptimizer(trainLossOp, learningRate)
    return trainUpdateOp

def getMeanSquareError(inputBatch, labelBatch, trainingPL, cnn):
    kernelSizes = [(GlobalOpts.kernelSize, ) * 3] * 3
    outputLayer = cnn(inputBatch,
                      trainingPL,
                      kernelSizes=kernelSizes,
                      strideSize=GlobalOpts.strideSize)
    return tf.losses.mean_squared_error(labels=labelBatch, predictions=outputLayer)

def GetModelOps(
        trainDataSet,
        valdDataSet,
        testDataSet,
        trainingPL,
        cnn):
    trainInputBatch, trainLabelBatch = trainDataSet.GetBatchOperations()
    trainLossOp = getMeanSquareError(trainInputBatch, trainLabelBatch, trainingPL, cnn)

    valdInputBatch, valdLabelBatch = valdDataSet.GetBatchOperations()
    valdLossOp = getMeanSquareError(valdInputBatch, valdLabelBatch, trainingPL, cnn)

    testInputBatch, testLabelBatch = testDataSet.GetBatchOperations()
    testLossOp = getMeanSquareError(testInputBatch, testLabelBatch, trainingPL, cnn)

    bootstrapInputBatch, bootstrapLabelBatch = testDataSet.GetRandomBatchOperations()
    bootstrapLossOp = getMeanSquareError(bootstrapInputBatch, bootstrapLabelBatch, trainingPL, cnn)

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
    learningRate = 0.0001
    trainLossOp, valdLossOp, testLossOp, bootstrapLossOp = \
        GetModelOps(
            trainDataSet,
            valdDataSet,
            testDataSet,
            trainingPL,
            GlobalOpts.cnn)

    trainUpdateOp = GetTrainingOperation(trainLossOp, learningRate)
    modelTrainer.DefineNewParams(GlobalOpts.summaryDir,
                                GlobalOpts.checkpointDir,
                                GlobalOpts.numSteps)

    config  = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GlobalOpts.gpuMemory
    with tf.Session(config=config) as sess:
        modelTrainer.RepeatTrials(sess,
                                  trainingPL,
                                  trainUpdateOp,
                                  trainLossOp,
                                  valdLossOp,
                                  testLossOp,
                                  name='{}{}'.format(GlobalOpts.concatType,
                                                     GlobalOpts.strideSize))

def CompareLearningRates(modelTrainer):
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    trainingPL = TrainingPlaceholder()
    learningRates = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    names = []; trainUpdateOps = [];
    trainLossOp, valdLossOp, testLossOp, bootstrapLossOp = \
        GetModelOps(
            trainDataSet,
            valdDataSet,
            testDataSet,
            trainingPL,
            GlobalOpts.cnn)
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
            'flag': '--strideSize',
            'help': 'The stride to chunk MRI images into. Typical values are 10, 15, 20, 30, 40, 60.',
            'action': 'store',
            'type': int,
            'dest': 'strideSize',
            'required': True
            },
            {
            'flag': '--concatType',
            'help': 'Concatenation type for patches. One of depth, batch, halfway.',
            'action': 'store',
            'type': str,
            'dest': 'concatType',
            'required': True
            }]
    ParseArgs('Run 3D CNN over structural MRI volumes', additionalArgs=additionalArgs)
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.DOWNSAMPLE_PATH')
    GlobalOpts.imageBatchDims = (-1, 61, 73, 61, 1)
    GlobalOpts.trainBatchSize = 4
    GlobalOpts.kernelSize = 3
    if GlobalOpts.concatType == 'depth':
        GlobalOpts.cnn = depthPatchCNN
    elif GlobalOpts.concatType == 'batch':
        GlobalOpts.cnn = batchPatchCNN
    elif GlobalOpts.concat = 'halfway':
        GlobalOpts.cnn = PatchCNN2Batch1Depth
    modelTrainer = ModelTrainer()

    GlobalOpts.summaryDir = '{}{}3D_stride{}/'.format(
                            get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'),
                            GlobalOpts.concatType,
                            GlobalOpts.strideSize)
    GlobalOpts.checkpointDir = '{}{}3D_stride{}/'.format(
                            get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'),
                            GlobalOpts.concatType,
                            GlobalOpts.strideSize)
    RunTestOnDirs(modelTrainer)
