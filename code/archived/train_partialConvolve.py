import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetNPY import DataSetNPY
from model.build_baselineStructuralCNN import *
from utils.saveModel import *
from utils.config import get
from engine.trainCommon import ModelTrainer
from placeholders.shared_placeholders import *

def GetTrainingOperation(lossOp, learningRate):
    with tf.variable_scope('optimizer'):
        updateOp = AdamOptimizer(lossOp, learningRate)
    return updateOp

def GetMSE(imagesPL, labelsPL, trainingPL, cnn):
    outputLayer = cnn(imagesPL,
                      trainingPL,
                      scale=GlobalOpts.scale)
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
    lossOp = GetMSE(imagesPL, labelsPL, trainingPL, GlobalOpts.cnn)
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
            'flag': '--scale',
            'help': 'The scale at which to slice dimensions. For example, a scale of 2 means that each dimension will be devided into 2 distinct regions, for a total of 8 contiguous chunks.',
            'action': 'store',
            'type': int,
            'dest': 'scale',
            'required': True
            },
            {
            'flag': '--type',
            'help': 'One of: depth, reverse, constant.',
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
    if GlobalOpts.type == 'depth':
        GlobalOpts.cnn = depthPatchCNN
    elif GlobalOpts.type == 'reverse':
        GlobalOpts.cnn = reverseDepthCNN
    elif GlobalOpts.type == 'constant':
        GlobalOpts.cnn = constantDepthCNN

    modelTrainer = ModelTrainer()

    GlobalOpts.summaryDir = '{}{}3D_scale{}_augment{}/'.format(
                            get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'),
                            GlobalOpts.type,
                            GlobalOpts.scale,
                            GlobalOpts.augment)
    GlobalOpts.checkpointDir = '{}{}3D_scale{}_augment{}/'.format(
                            get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'),
                            GlobalOpts.type,
                            GlobalOpts.scale,
                            GlobalOpts.augment)
    RunTestOnDirs(modelTrainer)
    
def compareDownsampling():
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
            'help': 'One of: depth, reverse, constant.',
            'action': 'store',
            'type': str,
            'dest': 'type',
            'required': True
            },
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
    GlobalOpts.name = '{}_scale{}_sample{}'.format(GlobalOpts.type, GlobalOpts.scale, GlobalOpts.downscaleRate)
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
    if GlobalOpts.type == 'depth':
        GlobalOpts.cnn = depthPatchCNN
    elif GlobalOpts.type == 'reverse':
        GlobalOpts.cnn = reverseDepthCNN
    elif GlobalOpts.type == 'constant':
        GlobalOpts.cnn = constantDepthCNN

    modelTrainer = ModelTrainer()

    GlobalOpts.summaryDir = '{}{}/'.format(get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'),
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '{}{}/'.format(get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'),
                                                     GlobalOpts.name)
    RunTestOnDirs(modelTrainer)
    
def compareSamplingType():
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
            'help': 'One of: depth, reverse',
            'action': 'store',
            'type': str,
            'dest': 'type',
            'required': True
            },
            {
            'flag': '--sampleType',
            'help': 'One of max, avg, sample, norm.',
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
    GlobalOpts.name = '{}_scale{}_{}'.format(GlobalOpts.type, GlobalOpts.scale, GlobalOpts.sampleType)
    GlobalOpts.imageBatchDims = (-1, 41, 49, 41, 1)
    if GlobalOpts.sampleType == 'max':
        GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.MAX_PATH')
    elif GlobalOpts.sampleType == 'avg':
        GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.AVG_PATH')
    elif GlobalOpts.sampleType == 'sample':
        GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.EXTRA_SMALL_PATH')
    elif GlobalOpts.sampleType == 'norm':
        GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.NORM_PATH')
        
    
    GlobalOpts.trainBatchSize = 4
    if GlobalOpts.type == 'depth':
        GlobalOpts.cnn = depthPatchCNN
    elif GlobalOpts.type == 'reverse':
        GlobalOpts.cnn = reverseDepthCNN

    modelTrainer = ModelTrainer()

    GlobalOpts.summaryDir = '{}{}/'.format('../summaries/sample_comp/',
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '{}{}/'.format('../summaries/sample_comp/',
                                                     GlobalOpts.name)
    RunTestOnDirs(modelTrainer)