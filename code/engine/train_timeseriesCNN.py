import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetNPY import DataSetNPY
from model.build_timeseriesCNN import TimeseriesCNN
from utils.saveModel import *
from utils.config import get
from engine.trainCommon import ModelTrainer
from placeholders.shared_placeholders import *

def GetTimeseriesCNN(
        trainDataSet,
        valdDataSet,
        testDataSet,
        trainingPL,
        learningRate=0.00005,
        keepProb=0.6):
    trainInputBatch, trainLabelBatch = trainDataSet.GetBatchOperations()
    trainOutputLayer = TimeseriesCNN(trainInputBatch,
                                trainingPL,
                                keepProbability=keepProb)
    trainLossOp = tf.losses.mean_squared_error(labels=trainLabelBatch, predictions=trainOutputLayer)
    with tf.variable_scope('optimizer'):
        trainUpdateOp = AdamOptimizer(trainLossOp, learningRate)

    valdInputBatch, valdLabelBatch = valdDataSet.GetBatchOperations()
    valdOutputLayer = TimeseriesCNN(valdInputBatch,
                               trainingPL,
                               keepProbability=keepProb)
    valdLossOp = tf.losses.mean_squared_error(labels=valdLabelBatch,
                                              predictions=valdOutputLayer)

    testInputBatch, testLabelBatch = testDataSet.GetBatchOperations()
    testOutputLayer = TimeseriesCNN(testInputBatch,
                               trainingPL,
                               keepProbability=keepProb)
    testLossOp = tf.losses.mean_squared_error(labels=testLabelBatch,
                                              predictions=testOutputLayer)

    return trainUpdateOp, trainLossOp, valdLossOp, testLossOp

def RunTestOnDirs(modelTrainer,
                  learningRate=0.00005,
                  keepProb=0.6):
    with tf.variable_scope(GlobalOpts.ModelScope):
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
        trainingPL = TrainingPlaceholder()
        trainUpdateOp, trainLossOp, valdLossOp, testLossOp = \
            GetTimeseriesCNN(
                trainDataSet,
                valdDataSet,
                testDataSet,
                trainingPL,
                learningRate=learningRate,
                keepProb=keepProb)
        modelTrainer.DefineNewParams(trainDataSet,
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
    ParseArgs('Run 1D convolutions over ROI timeseries data')
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.imageBaseString = get('DATA.TIMECOURSES.NUMPY_PATH')
    GlobalOpts.imageBatchDims = (-1, 120, 264)
    GlobalOpts.trainBatchSize = 32
    GlobalOpts.ModelScope = 'TimeseriesModel'
    GlobalOpts.axis = None

    modelTrainer = ModelTrainer()
    RunTestOnDirs(modelTrainer)
