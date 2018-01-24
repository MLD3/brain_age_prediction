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
from model.build_baselineStructuralCNN import baselineStructuralCNN, SliceCNN
from engine.train_baselineStructuralCNN import GetCNNBaselineModel
from utils import saveModel
from utils.config import get
from placeholders.shared_placeholders import *
from itertools import product
from engine.trainCommonNPY import *

def GetSliceCNN(imagesPL, trainingPL, labelsPL, learningRateName='LEARNING_RATE', stepCountName='NB_STEPS',
                        batchSizeName='BATCH_SIZE', keepProbName='KEEP_PROB', optimizer='ADAM', optionalHiddenLayerUnits=0,
                        downscaleRate=None):
    ############ DEFINE PLACEHOLDERS, LOSS ############
    predictionLayer = SliceCNN(imagesPL, trainingPL, keepProbability=get('TRAIN.CNN_BASELINE.%s' % keepProbName),
                                            optionalHiddenLayerUnits=optionalHiddenLayerUnits, downscaleRate=downscaleRate)
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

def GetXYZDataSet(PhenotypicsDF, useX=True, useY=True, useZ=True):
    assert useX or useY or useZ, 'Must choose at least one axis of slices for the data set'

    SubjectList = PhenotypicsDF['Subject'].tolist()
    LabelList = np.array(PhenotypicsDF['AgeYears'].tolist())

    labels = []
    fileList = []

    BaseDir = get('DATA.SLICES.BASE_DIR')
    xSlicesSuffix = get('DATA.SLICES.X_SLICES_DIR')
    ySlicesSuffix = get('DATA.SLICES.Y_SLICES_DIR')
    zSlicesSuffix = get('DATA.SLICES.Z_SLICES_DIR')

    for i in range(len(SubjectList)):
        subjectName = SubjectList[i]
        subjectLabel = LabelList[i]

        if useX:
            ###### READ X SLICES ######
            for j in range(121):
                fileName = '{}{}_x_{}.npy'.format(xSlicesSuffix, subjectName, j)
                fileList.append(fileName)
                labels.append(subjectLabel)

        if useY:
            ###### READ Y SLICES ######
            for j in range(145):
                fileName = '{}{}_y_{}.npy'.format(ySlicesSuffix, subjectName, j)
                fileList.append(fileName)
                labels.append(subjectLabel)

        if useZ:
            ###### READ Z SLICES ######
            for j in range(121):
                fileName = '{}{}_z_{}.npy'.format(zSlicesSuffix, subjectName, j)
                fileList.append(fileName)
                labels.append(subjectLabel)

    dataSet = DataSetNPY(numpyDirectory=BaseDir, numpyFileList=np.array(fileList), labels=np.array(labels))
    return

if __name__ == '__main__':
    PhenotypicsDF = readCSVData(get('DATA.PHENOTYPICS.PATH'))
    StructuralDataDir = get('DATA.STRUCTURAL.NUMPY_PATH')
    fileList = np.array([str(subject) + '.npy' for subject in PhenotypicsDF['Subject'].tolist()])
    labels = np.array(PhenotypicsDF['AgeYears'].tolist())

    NormalDataset = DataSetNPY(numpyDirectory=StructuralDataDir, numpyFileList=fileList, labels=labels)
    XYZDataset = GetXYZDataSet(PhenotypicsDF=PhenotypicsDF)
    XDataset = GetXYZDataSet(PhenotypicsDF=PhenotypicsDF, useX=True, useY=False, useZ=False)
    YDataset = GetXYZDataSet(PhenotypicsDF=PhenotypicsDF, useX=False, useY=True, useZ=False)
    ZDataset = GetXYZDataSet(PhenotypicsDF=PhenotypicsDF, useX=False, useY=False, useZ=True)

    dataSets = [NormalDataset, XYZDataset, XDataset, YDataset, ZDataset]

    trainingPL = TrainingPlaceholder()
    imagesPL, labelsPL = StructuralPlaceholders()
    slicesPL, _ = SlicePlaceholders()
    predictionLayers = []
    trainOperations = []
    lossFunctions = []
    stepCountArray = []
    batchSizeArray = []
    saveNames = []

    with tf.variable_scope('3DConvolutionStandard'):
        predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetCNNBaselineModel(imagesPL, trainingPL, labelsPL, batchSizeName='LARGE_BATCH_SIZE')
        predictionLayers.append(predictionLayer)
        trainOperations.append(trainOperation)
        lossFunctions.append(lossFunction)
        stepCountArray.append(stepCount)
        batchSizeArray.append(batchSize)
        saveNames.append(tf.contrib.framework.get_name_scope())

    with tf.variable_scope('SliceCNNAllAxes'):
        predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetSliceCNN(slicesPL, trainingPL, labelsPL, batchSizeName='SLICE_BATCH_SIZE')
        predictionLayers.append(predictionLayer)
        trainOperations.append(trainOperation)
        lossFunctions.append(lossFunction)
        stepCountArray.append(stepCount)
        batchSizeArray.append(batchSize)
        saveNames.append(tf.contrib.framework.get_name_scope())

    with tf.variable_scope('SliceCNNxAxis'):
        predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetSliceCNN(slicesPL, trainingPL, labelsPL, batchSizeName='SLICE_BATCH_SIZE')
        predictionLayers.append(predictionLayer)
        trainOperations.append(trainOperation)
        lossFunctions.append(lossFunction)
        stepCountArray.append(stepCount)
        batchSizeArray.append(batchSize)
        saveNames.append(tf.contrib.framework.get_name_scope())

    with tf.variable_scope('SliceCNNyAxis'):
        predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetSliceCNN(slicesPL, trainingPL, labelsPL, batchSizeName='SLICE_BATCH_SIZE')
        predictionLayers.append(predictionLayer)
        trainOperations.append(trainOperation)
        lossFunctions.append(lossFunction)
        stepCountArray.append(stepCount)
        batchSizeArray.append(batchSize)
        saveNames.append(tf.contrib.framework.get_name_scope())

    with tf.variable_scope('SliceCNNzAxis'):
        predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetSliceCNN(slicesPL, trainingPL, labelsPL, batchSizeName='SLICE_BATCH_SIZE')
        predictionLayers.append(predictionLayer)
        trainOperations.append(trainOperation)
        lossFunctions.append(lossFunction)
        stepCountArray.append(stepCount)
        batchSizeArray.append(batchSize)
        saveNames.append(tf.contrib.framework.get_name_scope())

    trainer = ModelTrainerNPY(summaryDir=get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'), checkpointDir=get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'))
    trainer.RunCrossValidation(dataSet, imagesPL, labelsPL, predictionLayers, trainOperations,
                                     lossFunctions, trainingPL, stepCountArray, batchSizeArray, saveNames)
