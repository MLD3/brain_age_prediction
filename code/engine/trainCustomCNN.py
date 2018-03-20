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
        updateOp, gradients = AdamOptimizer(lossOp, learningRate)
    return updateOp, gradients

def GetDataSetInputs():
    with tf.variable_scope('Inputs'):
        with tf.variable_scope('TrainingInputs'):
            trainDataSet = DataSetNPY(filenames=GlobalOpts.trainFiles,
                                      imageBaseString=GlobalOpts.imageBaseString,
                                      imageBatchDims=GlobalOpts.imageBatchDims,
                                      labelBaseString=GlobalOpts.labelBaseString,
                                      batchSize=GlobalOpts.trainBatchSize,
                                      augment=GlobalOpts.augment)
        with tf.variable_scope('ValidationInputs'):
            valdDataSet  = DataSetNPY(filenames=GlobalOpts.valdFiles,
                                    imageBaseString=GlobalOpts.imageBaseString,
                                    imageBatchDims=GlobalOpts.imageBatchDims,
                                    labelBaseString=GlobalOpts.labelBaseString,
                                    batchSize=1,
                                    maxItemsInQueue=GlobalOpts.numberValdItems,
                                    shuffle=False)
        with tf.variable_scope('TestInputs'):
            testDataSet  = DataSetNPY(filenames=GlobalOpts.testFiles,
                                    imageBaseString=GlobalOpts.imageBaseString,
                                    imageBatchDims=GlobalOpts.imageBatchDims,
                                    labelBaseString=GlobalOpts.labelBaseString,
                                    batchSize=1,
                                    maxItemsInQueue=GlobalOpts.numberTestItems,
                                    shuffle=False)
    return trainDataSet, valdDataSet, testDataSet

def DefineDataOpts(data='PNC', summaryName='test_comp'):
    if data == 'PNC':
        GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
        GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
        GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
        GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.AVG_PATH')
        GlobalOpts.labelBaseString = get('DATA.LABELS')
        GlobalOpts.imageBatchDims = (-1, 41, 49, 41, 1)
        GlobalOpts.trainBatchSize = 4
        GlobalOpts.numberTestItems = 100
        GlobalOpts.numberValdItems = 100
    elif data == 'ABIDE1':
        GlobalOpts.trainFiles = np.load(get('ABIDE.ABIDE1.TRAIN_LIST')).tolist()
        GlobalOpts.valdFiles = np.load(get('ABIDE.ABIDE1.VALD_LIST')).tolist()
        GlobalOpts.testFiles = np.load(get('ABIDE.ABIDE1.TEST_LIST')).tolist()
        GlobalOpts.imageBaseString = get('ABIDE.ABIDE1.AVG_POOL3')
        GlobalOpts.labelBaseString = get('ABIDE.ABIDE1.LABELS')
        GlobalOpts.imageBatchDims = (-1, 41, 49, 41, 1)
        GlobalOpts.trainBatchSize = 4
        GlobalOpts.numberTestItems = get('ABIDE.ABIDE1.NUM_TEST')
        GlobalOpts.numberValdItems = get('ABIDE.ABIDE1.NUM_VALD')
    elif data == 'ABIDE2':
        GlobalOpts.trainFiles = np.load(get('ABIDE.ABIDE2.TRAIN_LIST')).tolist()
        GlobalOpts.valdFiles = np.load(get('ABIDE.ABIDE2.VALD_LIST')).tolist()
        GlobalOpts.testFiles = np.load(get('ABIDE.ABIDE2.TEST_LIST')).tolist()
        GlobalOpts.imageBaseString = get('ABIDE.ABIDE2.AVG_POOL3')
        GlobalOpts.labelBaseString = get('ABIDE.ABIDE2.LABELS')
        GlobalOpts.imageBatchDims = (-1, 41, 49, 41, 1)
        GlobalOpts.trainBatchSize = 4
        GlobalOpts.numberTestItems = get('ABIDE.ABIDE2.NUM_TEST')
        GlobalOpts.numberValdItems = get('ABIDE.ABIDE2.NUM_VALD')
    GlobalOpts.poolType = 'MAX'
    GlobalOpts.name = '{}Scale{}Data{}'.format(GlobalOpts.type, GlobalOpts.scale, data)
    if GlobalOpts.sliceIndex is not None:
        GlobalOpts.name = '{}Slice{}'.format(GlobalOpts.name, GlobalOpts.sliceIndex)
    if GlobalOpts.align:
        GlobalOpts.name = '{}Aligned'.format(GlobalOpts.name)
    GlobalOpts.summaryDir = '../summaries/{}/{}/'.format(summaryName,
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '../checkpoints/{}/{}/'.format(summaryName,
                                                     GlobalOpts.name)
    GlobalOpts.augment = 'none'

def GetOps(labelsPL, outputLayer, learningRate=0.0001):
    if GlobalOpts.data == 'PNC':
        with tf.variable_scope('LossOperations'):
            lossOp = tf.losses.mean_squared_error(labels=labelsPL, predictions=outputLayer)
            MSEOp, MSEUpdateOp = tf.metrics.mean_squared_error(labels=labelsPL, predictions=outputLayer)
            MAEOp, MAEUpdateOp = tf.metrics.mean_absolute_error(labels=labelsPL, predictions=outputLayer)
            updateOp, gradients = GetTrainingOperation(lossOp, learningRate)
        printOps = PrintOps(ops=[MSEOp, MAEOp],
            updateOps=[MSEUpdateOp, MAEUpdateOp],
            names=['loss', 'MAE'],
            gradients=gradients)
    else:
        with tf.variable_scope('LossOperations'):
            oneHotLabels = tf.squeeze(tf.one_hot(indices=tf.cast(labelsPL, tf.int32), depth=2), axis=1)
            lossOp = tf.losses.softmax_cross_entropy(onehot_labels=oneHotLabels, logits=outputLayer)
            labelClasses = tf.argmax(input=oneHotLabels, axis=1)
            predictionClasses = tf.argmax(input=outputLayer, axis=1)
            accuracyOp, accuracyUpdateOp, = tf.metrics.accuracy(labels=labelClasses, predictions=predictionClasses)
            aucOp, aucUpdateOp = tf.metrics.auc(labels=labelClasses, predictions=predictionClasses)
            errorOp = 1.0 - accuracyOp
            updateOp, gradients = GetTrainingOperation(lossOp, learningRate)
        printOps = PrintOps(ops=[errorOp, aucOp],
            updateOps=[accuracyUpdateOp, aucUpdateOp],
            names=['loss', 'AUC'],
            gradients=gradients)
    return lossOp, printOps, updateOp

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
        },
        {
        'flag': '--summaryName',
        'help': 'The file name to put the results of this run into.',
        'action': 'store',
        'type': str,
        'dest': 'summaryName',
        'required': True
        },
        {
        'flag': '--data',
        'help': 'One of: PNC, ABIDE1, ABIDE2',
        'action': 'store',
        'type': str,
        'dest': 'data',
        'required': True
        },
        {
        'flag': '--sliceIndex',
        'help': 'Set this to an integer to select a single brain region as opposed to concatenating all regions along the depth channel.',
        'action': 'store',
        'type': int,
        'dest': 'sliceIndex',
        'required': False,
        'const': None
        },
        {
        'flag': '--align',
        'help': 'Set to true to align channels.',
        'action': 'store',
        'type': int,
        'dest': 'align',
        'required': False,
        'const': None
        },
        {
        'flag': '--numberTrials',
        'help': 'Number of repeated models to run.',
        'action': 'store',
        'type': int,
        'dest': 'numberTrials',
        'required': False,
        'const': None
        },
        ]
    ParseArgs('Run 3D CNN over structural MRI volumes', additionalArgs=additionalArgs)
    if GlobalOpts.numberTrials is None:
        GlobalOpts.numberTrials = 5
    DefineDataOpts(data=GlobalOpts.data, summaryName=GlobalOpts.summaryName)
    modelTrainer = ModelTrainer()
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    imagesPL, labelsPL = StructuralPlaceholders(GlobalOpts.imageBatchDims)
    trainingPL = TrainingPlaceholder()

    if GlobalOpts.type == 'traditional':
        convLayers = [8, 16, 32, 64]
    elif GlobalOpts.type == 'reverse':
        convLayers = [64, 32, 16, 8]
    if GlobalOpts.data == 'PNC':
        fullyConnectedLayers = [256, 1]
    else:
        fullyConnectedLayers = [256, 2]

    outputLayer = customCNN(imagesPL,
                            trainingPL,
                            GlobalOpts.scale,
                            convLayers,
                            fullyConnectedLayers,
                            poolType=GlobalOpts.poolType,
                            sliceIndex=GlobalOpts.sliceIndex,
                            align=GlobalOpts.align)
    lossOp, printOps, updateOp = GetOps(labelsPL, outputLayer)
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
                                  numIters=GlobalOpts.numberTrials)
