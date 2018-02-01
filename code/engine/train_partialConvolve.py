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
    if batchAxis == 'X':
        return tf.transpose(inputLayer, perm=[0, 2, 3, 1])
    elif batchAxis == 'Y':
        return tf.transpose(inputLayer, perm=[0, 1, 3, 2])
    elif batchAxis == 'Z':
        return inputLayer

def GetPartialConvolveCNN(
        trainDataSet,
        valdDataSet,
        testDataSet,
        trainingPL,
        learningRate=0.00005,
        batchAxis='X'):
    trainInputBatch, trainLabelBatch = trainDataSet.GetBatchOperations()
    trainInputBatch = ReshapeByAxis(trainInputBatch, batchAxis)
    trainOutputLayer = PartialConvolveCNN(trainInputBatch,
                                trainingPL)
    trainLossOp = tf.losses.mean_squared_error(labels=trainLabelBatch, predictions=trainOutputLayer)
    with tf.variable_scope('optimizer'):
        trainUpdateOp = AdamOptimizer(trainLossOp, learningRate)

    valdInputBatch, valdLabelBatch = valdDataSet.GetBatchOperations()
    valdInputBatch = ReshapeByAxis(valdInputBatch, batchAxis)
    valdOutputLayer = PartialConvolveCNN(valdInputBatch,
                               trainingPL)
    valdLossOp = tf.losses.mean_squared_error(labels=valdLabelBatch,
                                              predictions=valdOutputLayer)

    testInputBatch, testLabelBatch = testDataSet.GetBatchOperations()
    testInputBatch = ReshapeByAxis(testInputBatch, batchAxis)
    testOutputLayer = PartialConvolveCNN(testInputBatch,
                               trainingPL)
    testLossOp = tf.losses.mean_squared_error(labels=testLabelBatch,
                                              predictions=testOutputLayer)

    return trainUpdateOp, trainLossOp, valdLossOp, testLossOp

def RunTestOnDirs(modelTrainer,
                  learningRate=0.00005):
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
            GetPartialConvolveCNN(
                trainDataSet,
                valdDataSet,
                testDataSet,
                trainingPL,
                learningRate=learningRate)
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
    ParseArgs('Run 3D CNN over structural MRI volumes')
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.NUMPY_PATH')
    GlobalOpts.imageBatchDims = (-1, 121, 145, 121)
    GlobalOpts.trainBatchSize = 4
    GlobalOpts.ModelScope = '3DModel'
    GlobalOpts.axis = None

    modelTrainer = ModelTrainer()
    RunTestOnDirs(modelTrainer)
