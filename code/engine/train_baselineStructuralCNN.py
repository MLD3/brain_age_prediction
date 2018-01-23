import tensorflow as tf
import numpy as np
import pandas as pd
import math
import argparse
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from data_scripts.DataPlotter import PlotTrainingValidationLoss
from data_scripts.DataSetNPY import DataSetNPY
from sklearn.model_selection import train_test_split, KFold
from model.build_baselineStructuralCNN import baselineStructuralCNN
from utils import saveModel
from utils.config import get
from placeholders.shared_placeholders import *
from itertools import product
from engine.trainCommonNPY import *

def GetCNNBaselineModel(imagesPL, trainingPL, learningRateName='LEARNING_RATE', stepCountName='NB_STEPS',
                        batchSizeName='BATCH_SIZE', keepProbName='KEEP_PROB', optimizer='ADAM', optionalHiddenLayerUnits=0, useAttentionMap=False,
                        downscaleRate=None):
    ############ DEFINE PLACEHOLDERS, LOSS ############
    predictionLayer = baselineStructuralCNN(imagesPL, trainingPL, keepProbability=get('TRAIN.CNN_BASELINE.%s' % keepProbName),
                                            optionalHiddenLayerUnits=optionalHiddenLayerUnits, useAttentionMap=useAttentionMap, downscaleRate=downscaleRate)
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)

    ############ DEFINE OPTIMIZER ############
    if optimizer == 'ADAM':
        trainOperation = AdamOptimizer(lossFunction, get('TRAIN.CNN_BASELINE.%s' % learningRateName))
    elif optimizer == 'GRAD_DECAY':
        trainOperation = ScheduledGradOptimizer(lossFunction, baseLearningRate=get('TRAIN.CNN_BASELINE.%s' % learningRateName))

    ############ DEFINE LEARNING PARAMETERS ############
    stepCount = get('TRAIN.CNN_BASELINE.%s' % stepCountName)
    batchSize = get('TRAIN.CNN_BASELINE.%s' % batchSizeName)

    return predictionLayer, lossFunction, trainOperation, stepCount, batchSize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run 3D Convolution Scaling Tests on sMRI')
    parser.add_argument('--skewed', help='If specified, run skewed tests. Otherwise, run square tests.', action='store_true')
    args = parser.parse_args()

    PhenotypicsDF = readCSVData(get('DATA.PHENOTYPICS.PATH'))
    StructuralDataDir = get('DATA.STRUCTURAL.NUMPY_PATH')
    fileList = np.array([str(subject) + '.npy' for subject in PhenotypicsDF['Subject'].tolist()])
    labels = np.array(PhenotypicsDF['AgeYears'].tolist())
    dataSet = DataSetNPY(numpyDirectory=StructuralDataDir, numpyFileList=fileList, labels=labels)

    trainingPL = TrainingPlaceholder()
    imagesPL, labelsPL = StructuralPlaceholders()
    predictionLayers = []
    trainOperations = []
    lossFunctions = []
    stepCountArray = []
    batchSizeArray = []
    saveNames = []

    if not args.skewed:
        with tf.variable_scope('3DConvolutionStandard'):
            predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetCNNBaselineModel(imagesPL, trainingPL, batchSizeName='LARGE_BATCH_SIZE')
            predictionLayers.append(predictionLayer)
            trainOperations.append(trainOperation)
            lossFunctions.append(lossFunction)
            stepCountArray.append(stepCount)
            batchSizeArray.append(batchSize)
            saveNames.append(tf.contrib.framework.get_name_scope())

        with tf.variable_scope('3DConvolutionDownscale2x2x2'):
            predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetCNNBaselineModel(imagesPL, trainingPL, batchSizeName='LARGE_BATCH_SIZE', downscaleRate=2)
            predictionLayers.append(predictionLayer)
            trainOperations.append(trainOperation)
            lossFunctions.append(lossFunction)
            stepCountArray.append(stepCount)
            batchSizeArray.append(batchSize)
            saveNames.append(tf.contrib.framework.get_name_scope())

        with tf.variable_scope('3DConvolutionDownscale3x3x3'):
            predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetCNNBaselineModel(imagesPL, trainingPL, batchSizeName='LARGE_BATCH_SIZE', downscaleRate=3)
            predictionLayers.append(predictionLayer)
            trainOperations.append(trainOperation)
            lossFunctions.append(lossFunction)
            stepCountArray.append(stepCount)
            batchSizeArray.append(batchSize)
            saveNames.append(tf.contrib.framework.get_name_scope())

        with tf.variable_scope('3DConvolutionDownscale4x4x4'):
            predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetCNNBaselineModel(imagesPL, trainingPL, batchSizeName='LARGE_BATCH_SIZE', downscaleRate=4)
            predictionLayers.append(predictionLayer)
            trainOperations.append(trainOperation)
            lossFunctions.append(lossFunction)
            stepCountArray.append(stepCount)
            batchSizeArray.append(batchSize)
            saveNames.append(tf.contrib.framework.get_name_scope())
    else:
        with tf.variable_scope('3DConvolutionDownscale2x1x1'):
            predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetCNNBaselineModel(imagesPL, trainingPL, batchSizeName='LARGE_BATCH_SIZE', downscaleRate=[2,1,1])
            predictionLayers.append(predictionLayer)
            trainOperations.append(trainOperation)
            lossFunctions.append(lossFunction)
            stepCountArray.append(stepCount)
            batchSizeArray.append(batchSize)
            saveNames.append(tf.contrib.framework.get_name_scope())

        with tf.variable_scope('3DConvolutionDownscale1x2x1'):
            predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetCNNBaselineModel(imagesPL, trainingPL, batchSizeName='LARGE_BATCH_SIZE', downscaleRate=[1,2,1])
            predictionLayers.append(predictionLayer)
            trainOperations.append(trainOperation)
            lossFunctions.append(lossFunction)
            stepCountArray.append(stepCount)
            batchSizeArray.append(batchSize)
            saveNames.append(tf.contrib.framework.get_name_scope())

        with tf.variable_scope('3DConvolutionDownscale1x1x2'):
            predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetCNNBaselineModel(imagesPL, trainingPL, batchSizeName='LARGE_BATCH_SIZE', downscaleRate=[1,1,2])
            predictionLayers.append(predictionLayer)
            trainOperations.append(trainOperation)
            lossFunctions.append(lossFunction)
            stepCountArray.append(stepCount)
            batchSizeArray.append(batchSize)
            saveNames.append(tf.contrib.framework.get_name_scope())

    trainer = ModelTrainerNPY(summaryDir=get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'), checkpointDir=get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'))
    trainer.RunCrossValidation(dataSet, imagesPL, labelsPL, predictionLayers, trainOperations,
                                     lossFunctions, trainingPL, stepCountArray, batchSizeArray, saveNames)
