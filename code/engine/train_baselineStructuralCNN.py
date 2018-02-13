import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetNPY import DataSetNPY
from model.build_baselineStructuralCNN import baselineStructuralCNN
from utils.saveModel import *
from utils.config import get
from engine.trainCommon import ModelTrainer
from placeholders.shared_placeholders import *

def GetTrainingOperation(trainLossOp, learningRate):
    with tf.variable_scope('optimizer'):
        trainUpdateOp = AdamOptimizer(trainLossOp, learningRate)
    return trainUpdateOp

def getMeanSquareError(inputBatch, labelBatch, trainingPL):
    kernelSizes = [(GlobalOpts.kernelSize, ) * 3] * 3
    outputLayer = baselineStructuralCNN(inputBatch,
                      trainingPL,
                      kernelSizes=kernelSizes)
    return tf.losses.mean_squared_error(labels=labelBatch, predictions=outputLayer)

def GetStructuralCNN(
        trainDataSet,
        valdDataSet,
        testDataSet,
        trainingPL):
    kernelSizes = [(GlobalOpts.kernelSize, ) * 3] * 3
    trainInputBatch, trainLabelBatch = trainDataSet.GetBatchOperations()
    trainLossOp = getMeanSquareError(trainInputBatch, trainLabelBatch, trainingPL)

    valdInputBatch, valdLabelBatch = valdDataSet.GetBatchOperations()
    valdLossOp = getMeanSquareError(valdInputBatch, valdLabelBatch, trainingPL)

    testInputBatch, testLabelBatch = testDataSet.GetBatchOperations()
    testLossOp = getMeanSquareError(testInputBatch, testLabelBatch, trainingPL)

    bootstrapInputBatch, bootstrapLabelBatch = testDataSet.GetRandomBatchOperations()
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
    learningRates = [0.0001]
    names = []; trainUpdateOps = [];
    trainLossOp, valdLossOp, testLossOp, bootstrapLossOp = \
        GetStructuralCNN(
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
        modelTrainer.CompareRuns(sess, trainingPL, trainUpdateOps, trainLossOp, valdLossOp, testLossOp, names, None)

if __name__ == '__main__':
    ParseArgs('Run 3D CNN over structural MRI volumes')
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.DOWNSAMPLE_PATH')
    GlobalOpts.imageBatchDims = (-1, 61, 73, 61, 1)
    GlobalOpts.trainBatchSize = 4
    modelTrainer = ModelTrainer()

    GlobalOpts.kernelSize = 3
    GlobalOpts.summaryDir = get('TRAIN.CNN_BASELINE.SUMMARIES_DIR') + 'model3D_3x3x3/'
    GlobalOpts.checkpointDir = get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR') + 'model3D_3x3x3/'
    RunTestOnDirs(modelTrainer)
