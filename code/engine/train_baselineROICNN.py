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

def GetROIBaselineModel(learningRateName='LEARNING_RATE', stepCountName='NB_STEPS', batchSizeName='BATCH_SIZE'):
    ############ DEFINE PLACEHOLDERS, OPERATIONS ############
    trainingPL = TrainingPlaceholder()
    matricesPL, labelsPL = MatrixPlaceholders()
    predictionLayer = baselineROICNN(matricesPL, trainingPL)
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)

    trainOperation = AdamOptimizer(lossFunction, get('TRAIN.ROI_BASELINE.%s' % learningRateName))
    stepCount = get('TRAIN.ROI_BASELINE.%s' % stepCountName)
    batchSize = get('TRAIN.ROI_BASELINE.%s' % batchSizeName)

    return trainingPL, matricesPL, labelsPL, predictionLayer, lossFunction, trainOperation, stepCount, batchSize

if __name__ == '__main__':
    dataHolder = DataHolder(readCSVData(get('DATA.PHENOTYPICS.PATH')))
    dataHolder.getMatricesFromPath(get('DATA.MATRICES.PATH'))
    dataHolder.matricesToImages()
    dataSet = dataHolder.returnDataSet()

    matrixPlaceholders = []
    labelPlaceholders = []
    predictionLayers = []
    trainOperations = []
    lossFunctions = []
    trainingPlaceholders = []
    stepCountArray = []
    batchSizeArray = []
    saveNames = []

    trainingPL, matricesPL, labelsPL, predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetROIBaselineModel()
    matrixPlaceholders.append(matricesPL)
    labelPlaceholders.append(labelsPL)
    predictionLayers.append(predictionLayer)
    trainOperations.append(trainOperation)
    lossFunctions.append(lossFunction)
    trainingPlaceholders.append(trainingPL)
    stepCountArray.append(stepCount)
    batchSizeArray.append(batchSize)
    saveNames.append('DefaultSettings')

    trainingPL, matricesPL, labelsPL, predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetROIBaselineModel(learningRateName='LARGE_LEARNING_RATE')
    matrixPlaceholders.append(matricesPL)
    labelPlaceholders.append(labelsPL)
    predictionLayers.append(predictionLayer)
    trainOperations.append(trainOperation)
    lossFunctions.append(lossFunction)
    trainingPlaceholders.append(trainingPL)
    stepCountArray.append(stepCount)
    batchSizeArray.append(batchSize)
    saveNames.append('LargeLearningRate')

    trainingPL, matricesPL, labelsPL, predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetROIBaselineModel(batchSizeName='LARGE_BATCH_SIZE')
    matrixPlaceholders.append(matricesPL)
    labelPlaceholders.append(labelsPL)
    predictionLayers.append(predictionLayer)
    trainOperations.append(trainOperation)
    lossFunctions.append(lossFunction)
    trainingPlaceholders.append(trainingPL)
    stepCountArray.append(stepCount)
    batchSizeArray.append(batchSize)
    saveNames.append('LargeBatchSize')

    trainingPL, matricesPL, labelsPL, predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetROIBaselineModel(stepCountName='LARGE_NB_STEPS')
    matrixPlaceholders.append(matricesPL)
    labelPlaceholders.append(labelsPL)
    predictionLayers.append(predictionLayer)
    trainOperations.append(trainOperation)
    lossFunctions.append(lossFunction)
    trainingPlaceholders.append(trainingPL)
    stepCountArray.append(stepCount)
    batchSizeArray.append(batchSize)
    saveNames.append('LargeNumberOfIterations')

    RunCrossValidation(dataSet, matrixPlaceholders, labelPlaceholders, predictionLayers, trainOperations,
                                     lossFunctions, trainingPlaceholders, stepCountArray, batchSizeArray, saveNames)
