import tensorflow as tf
import numpy as np
import pandas as pd
import math
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from data_scripts.DataPlotter import PlotTrainingValidationLoss
from data_scripts.DataSet import DataSet
from sklearn.model_selection import train_test_split, KFold
from model.build_baselineROICNN import baselineROICNN
from utils import saveModel
from utils.config import get
from placeholders.shared_placeholders import *
from itertools import product
from engine.trainCommon import *

def GetROIBaselineModel(learningRateName='LEARNING_RATE', stepCountName='NB_STEPS',
                        batchSizeName='BATCH_SIZE', keepProbName='KEEP_PROB', optimizer='ADAM', firstHiddenLayerUnits=96, secondHiddenLayerUnits=0):
    ############ DEFINE PLACEHOLDERS, LOSS ############
    predictionLayer = baselineROICNN(matricesPL, trainingPL, keepProbability=get('TRAIN.ROI_BASELINE.%s' % keepProbName), firstHiddenLayerUnits=firstHiddenLayerUnits, secondHiddenLayerUnits=secondHiddenLayerUnits)
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)

    ############ DEFINE OPTIMIZER ############
    if optimizer == 'ADAM':
        trainOperation = AdamOptimizer(lossFunction, get('TRAIN.ROI_BASELINE.%s' % learningRateName))
    elif optimizer == 'GRAD_DECAY':
        trainOperation = ScheduledGradOptimizer(lossFunction, baseLearningRate=get('TRAIN.ROI_BASELINE.%s' % learningRateName))

    ############ DEFINE LEARNING PARAMETERS ############
    stepCount = get('TRAIN.ROI_BASELINE.%s' % stepCountName)
    batchSize = get('TRAIN.ROI_BASELINE.%s' % batchSizeName)

    return predictionLayer, lossFunction, trainOperation, stepCount, batchSize

if __name__ == '__main__':
    dataHolder = DataHolder(readCSVData(get('DATA.PHENOTYPICS.PATH')))
    dataHolder.getMatricesFromPath(get('DATA.MATRICES.PATH'))
    dataHolder.matricesToImages()
    dataSet = dataHolder.returnDataSet()
    trainingPL = TrainingPlaceholder()
    matricesPL, labelsPL = MatrixPlaceholders()

    predictionLayers = []
    trainOperations = []
    lossFunctions = []
    stepCountArray = []
    batchSizeArray = []
    saveNames = []

    # with tf.variable_scope('HeavyDropout'):
    #     predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetROIBaselineModel(keepProbName='SMALL_KEEP_PROB', stepCountName='LARGE_NB_STEPS')
    #     predictionLayers.append(predictionLayer)
    #     trainOperations.append(trainOperation)
    #     lossFunctions.append(lossFunction)
    #     stepCountArray.append(stepCount)
    #     batchSizeArray.append(batchSize)
    #     saveNames.append('HeavyDropout')
    #
    # with tf.variable_scope('ExtraHeavyDropout'):
    #     predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetROIBaselineModel(keepProbName='TINY_KEEP_PROB', stepCountName='LARGE_NB_STEPS')
    #     predictionLayers.append(predictionLayer)
    #     trainOperations.append(trainOperation)
    #     lossFunctions.append(lossFunction)
    #     stepCountArray.append(stepCount)
    #     batchSizeArray.append(batchSize)
    #     saveNames.append('ExtraHeavyDropout')
    #
    # with tf.variable_scope('ExtraHiddenLayer32Units'):
    #     predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetROIBaselineModel(keepProbName='SMALL_KEEP_PROB', stepCountName='LARGE_NB_STEPS', secondHiddenLayerUnits=32)
    #     predictionLayers.append(predictionLayer)
    #     trainOperations.append(trainOperation)
    #     lossFunctions.append(lossFunction)
    #     stepCountArray.append(stepCount)
    #     batchSizeArray.append(batchSize)
    #     saveNames.append('ExtraHiddenLayer32Units')
    #
    # with tf.variable_scope('HiddenLayer128Units'):
    #     predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetROIBaselineModel(keepProbName='SMALL_KEEP_PROB', stepCountName='LARGE_NB_STEPS', firstHiddenLayerUnits=128)
    #     predictionLayers.append(predictionLayer)
    #     trainOperations.append(trainOperation)
    #     lossFunctions.append(lossFunction)
    #     stepCountArray.append(stepCount)
    #     batchSizeArray.append(batchSize)
    #     saveNames.append('HiddenLayer128Units')
    #
    # with tf.variable_scope('TwoHiddenLayers_128_64_Units'):
    #     predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetROIBaselineModel(keepProbName='SMALL_KEEP_PROB', stepCountName='LARGE_NB_STEPS', firstHiddenLayerUnits=128, secondHiddenLayerUnits=64)
    #     predictionLayers.append(predictionLayer)
    #     trainOperations.append(trainOperation)
    #     lossFunctions.append(lossFunction)
    #     stepCountArray.append(stepCount)
    #     batchSizeArray.append(batchSize)
    #     saveNames.append('TwoHiddenLayers_128_64_Units')
    #
    # with tf.variable_scope('TwoHiddenLayersExtraHeavyDropout'):
    #     predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetROIBaselineModel(keepProbName='TINY_KEEP_PROB', stepCountName='LARGE_NB_STEPS', firstHiddenLayerUnits=128, secondHiddenLayerUnits=64)
    #     predictionLayers.append(predictionLayer)
    #     trainOperations.append(trainOperation)
    #     lossFunctions.append(lossFunction)
    #     stepCountArray.append(stepCount)
    #     batchSizeArray.append(batchSize)
    #     saveNames.append('TwoHiddenLayersExtraHeavyDropout')

    with tf.variable_scope('TwoHiddenLayersSmallDropout'):
        predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetROIBaselineModel(keepProbName='KEEP_PROB', stepCountName='SMALL_NB_STEPS', firstHiddenLayerUnits=128, secondHiddenLayerUnits=64)
        predictionLayers.append(predictionLayer)
        trainOperations.append(trainOperation)
        lossFunctions.append(lossFunction)
        stepCountArray.append(stepCount)
        batchSizeArray.append(batchSize)
        saveNames.append('TwoHiddenLayersSmallDropout')

    RunCrossValidation(dataSet, matricesPL, labelsPL, predictionLayers, trainOperations,
                                     lossFunctions, trainingPL, stepCountArray, batchSizeArray, saveNames)
