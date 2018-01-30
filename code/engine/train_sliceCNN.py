import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetBIN import DataSetBIN
from model.build_sliceCNN import SliceCNN
from utils.saveModel import *
from utils.config import get
from engine.trainCommonBIN import ModelTrainerBIN
from placeholders.shared_placeholders import *

def GetSliceCNN(
        trainDataSet,
        valdDataSet,
        testDataSet,
        trainingPL,
        learningRate=0.00005,
        keepProb=0.6,
        optionalHiddenLayerUnits=0,
        downscaleRate=None):
    trainInputBatch, trainLabelBatch = trainDataSet.GetBatchOperations()
    trainOutputLayer = SliceCNN(trainInputBatch,
                                trainingPL,
                                keepProbability=keepProb,
                                optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                                downscaleRate=downscaleRate)
    trainLossOp = tf.losses.mean_squared_error(labels=trainLabelBatch, predictions=trainOutputLayer)
    with tf.variable_scope('optimizer'):
        trainUpdateOp = AdamOptimizer(trainLossOp, learningRate)

    valdInputBatch, valdLabelBatch = valdDataSet.GetBatchOperations()
    valdOutputLayer = SliceCNN(valdInputBatch,
                               trainingPL,
                               keepProbability=keepProb,
                               optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                               downscaleRate=downscaleRate)
    valdLossOp = tf.losses.mean_squared_error(labels=valdLabelBatch,
                                              predictions=tf.reduce_mean(valdOutputLayer))

    testInputBatch, testLabelBatch = testDataSet.GetBatchOperations()
    testOutputLayer = SliceCNN(testInputBatch,
                               trainingPL,
                               keepProbability=keepProb,
                               optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                               downscaleRate=downscaleRate)
    testLossOp = tf.losses.mean_squared_error(labels=testLabelBatch,
                                              predictions=tf.reduce_mean(testOutputLayer))

    return trainUpdateOp, trainLossOp, valdLossOp, testLossOp

def RunTestOnDirs(modelTrainer,
                  learningRate=0.00005,
                  keepProb=0.6,
                  optionalHiddenLayerUnits=32):
    with tf.variable_scope(GlobalOpts.ModelScope):
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
        trainingPL = TrainingPlaceholder()
        trainUpdateOp, trainLossOp, valdLossOp, testLossOp = \
            GetSliceCNN(
                trainDataSet,
                valdDataSet,
                testDataSet,
                trainingPL,
                learningRate=learningRate,
                keepProb=keepProb,
                optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                downscaleRate=None)
        modelTrainer.DefineNewParams(saveName,
                                    trainDataSet,
                                    valdDataSet,
                                    testDataSet,
                                    GlobalOpts.summaryDir,
                                    GlobalOpts.checkpointDir,
                                    GlobalOpts.numSteps)
        WriteDefaultGraphToDir(dirName=GlobalOpts.summaryDir)
        config  = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = GlobalOpts.gpuMemory
        with tf.Session(config=config) as sess:
            modelTrainer.TrainModel(sess, trainingPL, trainUpdateOp, trainLossOp, valdLossOp, testLossOp)

if __name__ == '__main__':
    ParseArgs('Run 2D convolution tests over structural MRI slices', data=True)
    GlobalOpts.valdFiles = [get('DATA.BIN.VALD')]
    GlobalOpts.testFiles = [get('DATA.BIN.TEST')]
    GlobalOpts.trainImageDims = [145, 145, 1]
    GlobalOpts.testImageDims = [121, 145, 121, 1]
    GlobalOpts.trainBatchSize = 64

    modelTrainer = ModelTrainerBIN()

    if GlobalOpts.data == 'X':
        GlobalOpts.ModelScope = 'xAxisModel'
        GlobalOpts.trainFiles = [get('DATA.BIN.X_SLICES_TRAIN')]
        GlobalOpts.axis = 0
    elif GlobalOpts.data == 'Y':
        GlobalOpts.ModelScope = 'yAxisModel'
        GlobalOpts.trainFiles = [get('DATA.BIN.Y_SLICES_TRAIN')]
        GlobalOpts.axis = 1
    elif GlobalOpts.data == 'Z':
        GlobalOpts.ModelScope = 'zAxisModel'
        GlobalOpts.trainFiles = [get('DATA.BIN.Z_SLICES_TRAIN')]
        GlobalOpts.axis = 2
    elif GlobalOpts.data == 'XYZ':
        GlobalOpts.ModelScope = 'xyzAxisModel'
        GlobalOpts.trainFiles = [get('DATA.BIN.X_SLICES_TRAIN'),
                                 get('DATA.BIN.Y_SLICES_TRAIN'),
                                 get('DATA.BIN.Z_SLICES_TRAIN')]
        GlobalOpts.axis = 3
    else:
        print('Unrecognized data argument {}.'.format(args.data))

    RunTestOnDirs(modelTrainer)
