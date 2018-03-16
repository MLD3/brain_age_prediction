import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetNPY import DataSetNPY
from model.buildCustomCNN import customCNN
from utils.saveModel import *
from utils.config import get
from engine.trainCommon import *
from placeholders.shared_placeholders import *

def GetTrainingOperation(lossOp, learningRate):
    with tf.variable_scope('optimizer'):
        updateOp = AdamOptimizer(lossOp, learningRate)
    return updateOp


def GetDataSetInputs():
    with tf.variable_scope('Inputs'):
        with tf.variable_scope('TrainingInputs'):
            trainDataSet = DataSetNPY(filenames=GlobalOpts.trainFiles,
                                      imageBaseString=GlobalOpts.imageBaseString,
                                      imageBatchDims=GlobalOpts.imageBatchDims,
                                      batchSize=GlobalOpts.trainBatchSize,
                                      augment=GlobalOpts.augment)
        with tf.variable_scope('ValidationInputs'):
            valdDataSet  = DataSetNPY(filenames=GlobalOpts.valdFiles,
                                    imageBaseString=GlobalOpts.imageBaseString,
                                    imageBatchDims=GlobalOpts.imageBatchDims,
                                    batchSize=1,
                                    maxItemsInQueue=GlobalOpts.numberValdItems,
                                    shuffle=False)
        with tf.variable_scope('TestInputs'):
            testDataSet  = DataSetNPY(filenames=GlobalOpts.testFiles,
                                    imageBaseString=GlobalOpts.imageBaseString,
                                    imageBatchDims=GlobalOpts.imageBatchDims,
                                    batchSize=1,
                                    maxItemsInQueue=GlobalOpts.numberTestItems,
                                    shuffle=False)
    return trainDataSet, valdDataSet, testDataSet

def compareCustomCNN():
    additionalArgs = [
        {
        'flag': '--scale',
        'help': 'The scale at which to slice dimensions. For example, a scale of 2 means that each dimension will be devided into 2 distinct regions, for a total of 8 contiguous chunks.',
        'action': 'store',
        'type': int,
        'dest': 'scale',
        'required': True
        },
        {
        'flag': '--type',
        'help': 'One of: traditional, reverse',
        'action': 'store',
        'type': str,
        'dest': 'type',
        'required': True
        }
        ]
    ParseArgs('Run 3D CNN over structural MRI volumes', additionalArgs=additionalArgs)
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.NORM_PATH')
    GlobalOpts.imageBatchDims = (-1, 41, 49, 41, 1)
    GlobalOpts.trainBatchSize = 4
    GlobalOpts.augment = 'none'
    GlobalOpts.name = '{}Scale{}'.format(GlobalOpts.type, GlobalOpts.scale)
    summaryName = 'test_comp'
    GlobalOpts.summaryDir = '../summaries/{}/{}/'.format(summaryName,
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '../checkpoints/{}/{}/'.format(summaryName,
                                                     GlobalOpts.name)
    GlobalOpts.numberTestItems = 75
    GlobalOpts.numberValdItems = 75
    GlobalOpts.poolType = 'AVERAGE'

    modelTrainer = ModelTrainer()
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    imagesPL, labelsPL = StructuralPlaceholders(GlobalOpts.imageBatchDims)
    trainingPL = TrainingPlaceholder()

    fullyConnectedLayers = [256, 1]
    if GlobalOpts.type == 'traditional':
        convLayers = [8, 16, 32, 64]
    elif GlobalOpts.type == 'reverse':
        convLayers = [64, 32, 16, 8]

    outputLayer = customCNN(imagesPL,
                            trainingPL,
                            GlobalOpts.scale,
                            convLayers,
                            fullyConnectedLayers,
                            poolType=GlobalOpts.poolType)
    lossOp = tf.losses.mean_squared_error(labels=labelsPL, predictions=outputLayer)
    MAEOp = tf.reduce_mean(tf.abs(tf.subtract(labelsPL, outputLayer)))
    printOps = PrintOps(ops=[lossOp, MAEOp], names=['loss', 'MAE'])

    learningRate = 0.0001
    updateOp = GetTrainingOperation(lossOp, learningRate)
    modelTrainer.DefineNewParams(GlobalOpts.summaryDir,
                                GlobalOpts.checkpointDir,
                                imagesPL,
                                trainingPL,
                                labelsPL,
                                trainDataSet,
                                valdDataSet,
                                testDataSet,
                                GlobalOpts.numSteps)
    config  = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GlobalOpts.gpuMemory
    with tf.Session(config=config) as sess:
        modelTrainer.RepeatTrials(sess,
                                  updateOp,
                                  printOps,
                                  name=GlobalOpts.name,
                                  numIters=5)
