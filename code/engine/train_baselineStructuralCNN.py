import tensorflow as tf
import numpy as np
import pandas as pd
import math
from utils.args import *
from data_scripts.DataReader import *
from data_scripts.DataPlotter import PlotTrainingValidationLoss
from data_scripts.DataSetNPY import DataSetNPY
from sklearn.model_selection import train_test_split, KFold
from model.build_baselineStructuralCNN import baselineStructuralCNN
from utils.saveModel import *
from utils.config import get
from placeholders.shared_placeholders import *
from itertools import product
from engine.trainCommonNPY import *

def GetStructuralCNN(
        trainDataSet,
        valdDataSet,
        testDataSet,
        trainingPL,
        learningRate=0.00005,
        keepProb=0.6,
        optionalHiddenLayerUnits=0,
        downscaleRate=None):
    trainInputBatch, trainLabelBatch = trainDataSet.GetBatchOperations()
    trainOutputLayer = baselineStructuralCNN(trainInputBatch,
                                trainingPL,
                                keepProbability=keepProb,
                                optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                                downscaleRate=downscaleRate)
    trainLossOp = tf.losses.mean_squared_error(labels=trainLabelBatch, predictions=trainOutputLayer)
    with tf.variable_scope('optimizer'):
        trainUpdateOp = AdamOptimizer(trainLossOp, learningRate)

    valdInputBatch, valdLabelBatch = valdDataSet.GetBatchOperations()
    valdOutputLayer = baselineStructuralCNN(valdInputBatch,
                               trainingPL,
                               keepProbability=keepProb,
                               optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                               downscaleRate=downscaleRate)
    valdLossOp = tf.losses.mean_squared_error(labels=valdLabelBatch,
                                              predictions=valdOutputLayer)

    testInputBatch, testLabelBatch = testDataSet.GetBatchOperations()
    testOutputLayer = baselineStructuralCNN(testInputBatch,
                               trainingPL,
                               keepProbability=keepProb,
                               optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                               downscaleRate=downscaleRate)
    testLossOp = tf.losses.mean_squared_error(labels=testLabelBatch,
                                              predictions=testOutputLayer)

    return trainUpdateOp, trainLossOp, valdLossOp, testLossOp

def RunTestOnDirs(modelTrainer,
                  learningRate=0.00005,
                  keepProb=0.6,
                  optionalHiddenLayerUnits=0):
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
                                    shuffle=False)
        with tf.variable_scope('TestInputs'):
            testDataSet  = DataSetBIN(binFileNames=GlobalOpts.testFiles,
                                    imageDims=GlobalOpts.testImageDims,
                                    batchSize=1,
                                    maxItemsInQueue=75,
                                    minItemsInQueue=1,
                                    shuffle=False)
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
    ParseArgs()
    GlobalOpts.trainFiles = [get('DATA.BIN.TRAIN')]
    GlobalOpts.valdFiles = [get('DATA.BIN.VALD')]
    GlobalOpts.testFiles = [get('DATA.BIN.TEST')]
    GlobalOpts.trainImageDims = [121, 145, 121, 1]
    GlobalOpts.testImageDims = [121, 145, 121, 1]
    GlobalOpts.trainBatchSize = 4
    GlobalOpts.ModelScope = '3DModel'
    GlobalOpts.axis = None

    modelTrainer = ModelTrainerBIN()
    RunTestOnDirs(modelTrainer)
