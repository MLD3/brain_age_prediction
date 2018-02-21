import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetNPY import DataSetNPY
from model.build_baselineStructuralCNN import baselineStructuralCNN, reverseBaseline, constantBaseline
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
                                  name='{}baseline3D'.format(GlobalOpts.type),
                                  numIters=5)

if __name__ == '__main__':
    additionalArgs = [
            {
            'flag': '--type',
            'help': 'One of: standard, reverse, constant.',
            'action': 'store',
            'type': str,
            'dest': 'type',
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

    GlobalOpts.summaryDir = '{}{}baseline3D/'.format(get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'), GlobalOpts.type)
    GlobalOpts.checkpointDir = '{}{}baseline3D/'.format(get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'), GlobalOpts.type)
    RunTestOnDirs(modelTrainer)
