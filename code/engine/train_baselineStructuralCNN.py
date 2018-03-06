import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetNPY import DataSetNPY
from model.build_baselineStructuralCNN import baselineStructuralCNN, reverseBaseline, constantBaseline, customCNN, attentionMapCNN, deepCNN
from utils.saveModel import *
from utils.config import get
from engine.trainCommon import ModelTrainer
from placeholders.shared_placeholders import *

def GetTrainingOperation(lossOp, learningRate):
    with tf.variable_scope('optimizer'):
        updateOp = AdamOptimizer(lossOp, learningRate)
    return updateOp

def GetMSE(imagesPL, labelsPL, trainingPL):
    outputLayer = GlobalOpts.cnn(imagesPL,
                      trainingPL)
    return tf.losses.mean_squared_error(labels=labelsPL, predictions=outputLayer)

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
    imagesPL, labelsPL = StructuralPlaceholders(GlobalOpts.imageBatchDims)
    trainingPL = TrainingPlaceholder()
    lossOp = GetMSE(imagesPL, labelsPL, trainingPL)
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
                                  lossOp,
                                  name=GlobalOpts.name,
                                  numIters=5)
def compareAugmentations():
    additionalArgs = [
            {
            'flag': '--type',
            'help': 'One of: standard, reverse, constant.',
            'action': 'store',
            'type': str,
            'dest': 'type',
            'required': True
            },
            {
            'flag': '--augment',
            'help': 'One of: none, translate, flip.',
            'action': 'store',
            'type': str,
            'dest': 'augment',
            'required': True
            }]
    ParseArgs('Run 3D CNN over structural MRI volumes', additionalArgs=additionalArgs)
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.DOWNSAMPLE_PATH')
    GlobalOpts.imageBatchDims = (-1, 61, 73, 61, 1)
    # GlobalOpts.imageBatchDims = (-1, 121, 145, 121, 1)
    GlobalOpts.trainBatchSize = 4
    if GlobalOpts.type == 'standard':
        GlobalOpts.cnn = baselineStructuralCNN
    elif GlobalOpts.type == 'reverse':
        GlobalOpts.cnn = reverseBaseline
    elif GlobalOpts.type == 'constant':
        GlobalOpts.cnn = constantBaseline
    modelTrainer = ModelTrainer()

    GlobalOpts.summaryDir = '{}{}baseline3D_augment{}/'.format(get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'),
                                                     GlobalOpts.type,
                                                     GlobalOpts.augment)
    GlobalOpts.checkpointDir = '{}{}baseline3D_augment{}/'.format(get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'),
                                                     GlobalOpts.type,
                                                     GlobalOpts.augment)
    RunTestOnDirs(modelTrainer)

def compareDownsampling():
    additionalArgs = [
        {
        'flag': '--downscaleRate',
        'help': 'One of 1, 2, 3.',
        'action': 'store',
        'type': int,
        'dest': 'downscaleRate',
        'required': True
        }
    ]
    ParseArgs('Run 3D CNN over structural MRI volumes', additionalArgs=additionalArgs)
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.augment = 'none'
    GlobalOpts.name = 'baseline3D_scale{}'.format(GlobalOpts.downscaleRate)
    if GlobalOpts.downscaleRate == 1:
        GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.NUMPY_PATH')
        GlobalOpts.imageBatchDims = (-1, 121, 145, 121, 1)
    elif GlobalOpts.downscaleRate == 2:
        GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.DOWNSAMPLE_PATH')
        GlobalOpts.imageBatchDims = (-1, 61, 73, 61, 1)
    elif GlobalOpts.downscaleRate == 3:
        GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.EXTRA_SMALL_PATH')
        GlobalOpts.imageBatchDims = (-1, 41, 49, 41, 1)
    
    GlobalOpts.trainBatchSize = 4
    GlobalOpts.cnn = baselineStructuralCNN
    modelTrainer = ModelTrainer()

    GlobalOpts.summaryDir = '{}{}/'.format(get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'),
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '{}{}/'.format(get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'),
                                                     GlobalOpts.name)
    RunTestOnDirs(modelTrainer)
    
def compareSamplingType():
    additionalArgs = [
        {
        'flag': '--sampleType',
        'help': 'One of max, avg, sample.',
        'action': 'store',
        'type': str,
        'dest': 'sampleType',
        'required': True
        }
    ]
    ParseArgs('Run 3D CNN over structural MRI volumes', additionalArgs=additionalArgs)
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.augment = 'none'
    GlobalOpts.name = 'baseline3D_{}'.format(GlobalOpts.sampleType)
    GlobalOpts.imageBatchDims = (-1, 41, 49, 41, 1)
    if GlobalOpts.sampleType == 'max':
        GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.MAX_PATH')
    elif GlobalOpts.sampleType == 'avg':
        GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.AVG_PATH')
    elif GlobalOpts.sampleType == 'sample':
        GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.EXTRA_SMALL_PATH')
    
    GlobalOpts.trainBatchSize = 4
    GlobalOpts.cnn = baselineStructuralCNN
    modelTrainer = ModelTrainer()

    GlobalOpts.summaryDir = '{}{}/'.format('../summaries/sample_comp/',
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '{}{}/'.format('../summaries/sample_comp/',
                                                     GlobalOpts.name)
    RunTestOnDirs(modelTrainer)

def GetCustomMSE(imagesPL, labelsPL, trainingPL, convLayers, fullyConnectedLayers):
    outputLayer = customCNN(imagesPL,
                            trainingPL,
                            GlobalOpts.strideSize, 
                            convLayers,
                            fullyConnectedLayers)
    return tf.losses.mean_squared_error(labels=labelsPL, predictions=outputLayer)

def compareArchitectures():
    additionalArgs = [
        {
        'flag': '--strideSize',
        'help': 'The stride to chunk MRI images into. Typical values are 10, 15, 20, 30, 40, 60.',
        'action': 'store',
        'type': int,
        'dest': 'strideSize',
        'required': True
        },
        {
        'flag': '--type',
        'help': 'One of: traditional, reverse, hourglass, diamond',
        'action': 'store',
        'type': str,
        'dest': 'type',
        'required': True
        }]
    ParseArgs('Run 3D CNN over structural MRI volumes', additionalArgs=additionalArgs)
    if GlobalOpts.strideSize <= 0:
        GlobalOpts.strideSize = None
        GlobalOpts.cnn = baselineStructuralCNN
        
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.DOWNSAMPLE_PATH')
    GlobalOpts.imageBatchDims = (-1, 61, 73, 61, 1)
    GlobalOpts.trainBatchSize = 4
    GlobalOpts.augment = 'none'
    GlobalOpts.name = '{}_stride{}'.format(GlobalOpts.type, GlobalOpts.strideSize)
    modelTrainer = ModelTrainer()
    GlobalOpts.summaryDir = '{}{}/'.format(get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'),
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '{}{}/'.format(get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'),
                                                     GlobalOpts.name)
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    imagesPL, labelsPL = StructuralPlaceholders(GlobalOpts.imageBatchDims)
    trainingPL = TrainingPlaceholder()
    
    fullyConnectedLayers = [256]
    if GlobalOpts.type == 'traditional':
        convLayers = [4, 8, 16]
    elif GlobalOpts.type == 'reverse':
        convLayers = [16, 8, 4]
    elif GlobalOpts.type == 'hourglass':
        convLayers = [12, 4, 12]
    elif GlobalOpts.type == 'diamond':
        convLayers = [6, 16, 6]
        
    lossOp = GetCustomMSE(imagesPL, labelsPL, trainingPL, convLayers, fullyConnectedLayers)
    
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
                                  lossOp,
                                  name=GlobalOpts.name,
                                  numIters=5)
        
def comparePools():
    additionalArgs = [
        {
        'flag': '--strideSize',
        'help': 'The stride to chunk MRI images into. Typical values are 10, 15, 20, 30, 40, 60.',
        'action': 'store',
        'type': int,
        'dest': 'strideSize',
        'required': True
        },
        {
        'flag': '--poolType',
        'help': 'One of: MAX, AVERAGE, none',
        'action': 'store',
        'type': str,
        'dest': 'poolType',
        'required': True
        },
        {
        'flag': '--poolStride',
        'help': 'The stride length to use for pooling, 1 or 2.',
        'action': 'store',
        'type': int,
        'dest': 'poolStride',
        'required': True
        },
        {
        'flag': '--convStride',
        'help': 'The stride length to use for convolutions, 1 or 2.',
        'action': 'store',
        'type': int,
        'dest': 'convStride',
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
    if GlobalOpts.strideSize <= 0:
        GlobalOpts.strideSize = None
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.DOWNSAMPLE_PATH')
    GlobalOpts.imageBatchDims = (-1, 61, 73, 61, 1)
    GlobalOpts.trainBatchSize = 4
    GlobalOpts.augment = 'none'
    GlobalOpts.name = '{}_slice{}_pool{}{}_conv{}'.format(GlobalOpts.type, 
                                                          GlobalOpts.strideSize,
                                                          GlobalOpts.poolType,
                                                          GlobalOpts.poolStride,
                                                          GlobalOpts.convStride)
    modelTrainer = ModelTrainer()
    GlobalOpts.summaryDir = '{}{}/'.format('../summaries/pool_comp/',
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '{}{}/'.format('../checkpoints/pool_comp/',
                                                     GlobalOpts.name)
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    imagesPL, labelsPL = StructuralPlaceholders(GlobalOpts.imageBatchDims)
    trainingPL = TrainingPlaceholder()
    
    fullyConnectedLayers = [256]
    
    if GlobalOpts.type == 'traditional':
        convLayers = [8, 16, 32]
    elif GlobalOpts.type == 'reverse':
        convLayers = [32, 16, 8]
        
    outputLayer = customCNN(imagesPL,
                            trainingPL,
                            GlobalOpts.strideSize, 
                            convLayers,
                            fullyConnectedLayers,
                            convStrides=(GlobalOpts.convStride, ) * 4,
                            poolStrides=(GlobalOpts.poolStride, ) * 4,
                            poolType=GlobalOpts.poolType)
    lossOp = tf.losses.mean_squared_error(labels=labelsPL, predictions=outputLayer)
    
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
                                  lossOp,
                                  name=GlobalOpts.name,
                                  numIters=5)

def compareAttention():
    additionalArgs = [
        {
        'flag': '--strideSize',
        'help': 'The stride to chunk MRI images into. Typical values are 10, 15, 20, 30, 40, 60.',
        'action': 'store',
        'type': int,
        'dest': 'strideSize',
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
        'flag': '--attention',
        'help': 'One of: 0, 1, 2, 3',
        'action': 'store',
        'type': int,
        'dest': 'attention',
        'required': True
        }
        ]
    ParseArgs('Run 3D CNN over structural MRI volumes', additionalArgs=additionalArgs)
    if GlobalOpts.strideSize <= 0:
        GlobalOpts.strideSize = None
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.DOWNSAMPLE_PATH')
    GlobalOpts.imageBatchDims = (-1, 61, 73, 61, 1)
    GlobalOpts.trainBatchSize = 4
    GlobalOpts.augment = 'none'
    GlobalOpts.name = 'attention{}_{}_stride{}'.format(GlobalOpts.attention, 
                                                       GlobalOpts.type, 
                                                       GlobalOpts.strideSize)
    modelTrainer = ModelTrainer()
    GlobalOpts.summaryDir = '{}{}/'.format('../summaries/attention_comp/',
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '{}{}/'.format('../checkpoints/attention_comp/',
                                                     GlobalOpts.name)
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    imagesPL, labelsPL = StructuralPlaceholders(GlobalOpts.imageBatchDims)
    trainingPL = TrainingPlaceholder()
        
    if GlobalOpts.type == 'traditional':
        convLayers = [8, 16, 32, 64]
    elif GlobalOpts.type == 'reverse':
        convLayers = [64, 32, 16, 8]
    
    randomAttentionStarter=False
    if GlobalOpts.attention == 0:
        attentionMapBools=[True, False, False, False]
    elif GlobalOpts.attention == 1:
        attentionMapBools=[False, True, False, False]
    elif GlobalOpts.attention == 2:
        attentionMapBools=[True, True, False, False]
    elif GlobalOpts.attention == 3:
        attentionMapBools=[False, False, False, False]
        randomAttentionStarter=True
        
    outputLayer = attentionMapCNN(imagesPL,
                            trainingPL,
                            GlobalOpts.strideSize, 
                            convLayers,
                            attentionMapBools,
                            randomAttentionStarter=randomAttentionStarter)
    lossOp = tf.losses.mean_squared_error(labels=labelsPL, predictions=outputLayer)
    
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
                                  lossOp,
                                  name=GlobalOpts.name,
                                  numIters=5)

def compareDeep():
    additionalArgs = [
        {
        'flag': '--strideSize',
        'help': 'The stride to chunk MRI images into. Typical values are 10, 15, 20, 30, 40, 60.',
        'action': 'store',
        'type': int,
        'dest': 'strideSize',
        'required': True
        },
        {
        'flag': '--reverse',
        'help': '0 (False) or 1 (True).',
        'action': 'store',
        'type': int,
        'dest': 'reverse',
        'required': True
        }]
    ParseArgs('Run 3D CNN over structural MRI volumes', additionalArgs=additionalArgs)
    if GlobalOpts.strideSize <= 0:
        GlobalOpts.strideSize = None
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
    GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.DOWNSAMPLE_PATH')
    GlobalOpts.imageBatchDims = (-1, 61, 73, 61, 1)
    GlobalOpts.trainBatchSize = 4
    GlobalOpts.augment = 'none'
    GlobalOpts.reverse = (GlobalOpts.reverse == 1)
    GlobalOpts.name = 'deep_stride{}_reverse{}'.format(GlobalOpts.strideSize, 
                                                       GlobalOpts.reverse)
    modelTrainer = ModelTrainer()
    GlobalOpts.summaryDir = '{}{}/'.format('../summaries/deep_comp/',
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '{}{}/'.format('../checkpoints/deep_comp/',
                                                     GlobalOpts.name)
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    imagesPL, labelsPL = StructuralPlaceholders(GlobalOpts.imageBatchDims)
    trainingPL = TrainingPlaceholder()

    outputLayer = deepCNN(imagesPL,
                            trainingPL,
                            GlobalOpts.strideSize, 
                            reverse=GlobalOpts.reverse)
    lossOp = tf.losses.mean_squared_error(labels=labelsPL, predictions=outputLayer)
    
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
                                  lossOp,
                                  name=GlobalOpts.name,
                                  numIters=5)