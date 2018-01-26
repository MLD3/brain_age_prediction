import tensorflow as tf
import numpy as np
import pandas as pd
import math
import argparse
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from data_scripts.DataPlotter import PlotTrainingValidationLoss
from data_scripts.DataSetBIN import DataSetBIN
from sklearn.model_selection import train_test_split, KFold
from model.build_baselineStructuralCNN import baselineStructuralCNN, SliceCNN
from engine.train_baselineStructuralCNN import GetCNNBaselineModel
from utils import saveModel
from utils.config import get
from placeholders.shared_placeholders import *
from itertools import product
from engine.trainCommonBIN import *

def GetSliceCNN(
        trainingDataSet,
        validationDataSet,
        testDataSet,
        trainingPL,
        learningRateName='LEARNING_RATE',
        keepProbName='KEEP_PROB',
        optionalHiddenLayerUnits=0,
        downscaleRate=None):
    trainInputBatch, trainLabelBatch = trainingDataSet.GetBatchOperations()
    trainOutputLayer = SliceCNN(trainInputBatch,
                                trainingPL,
                                keepProbability=get('TRAIN.CNN_BASELINE.%s' % keepProbName),
                                optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                                downscaleRate=downscaleRate)
    trainLossOp = tf.losses.mean_squared_error(labels=trainLabelBatch, predictions=trainOutputLayer)
    trainUpdateOp = AdamOptimizer(lossFunction, get('TRAIN.CNN_BASELINE.%s' % learningRateName))

    valdInputBatch, valdLabelBatch = validationDataSet.GetBatchOperations()
    valdOutputLayer = SliceCNN(valdInputBatch,
                               trainingPL,
                               keepProbability=get('TRAIN.CNN_BASELINE.%s' % keepProbName),
                               optionalHiddenLayerUnits=optionalHiddenLayerUnits,
                               downscaleRate=downscaleRate)
    valdLossOp = tf.losses.mean_squared_error(labels=valdLabelBatch, predictions=valdOutputLayer)
    return trainUpdateOp, trainLossOp, valdLossOp

def RunTestOnDirs(modelTrainer, trainFiles, valdFiles, testFiles):
    tf.reset_default_graph()
    trainDataSet = DataSetBIN(binFileNames=trainFiles)
    valdDataSet  = DataSetBIN(binFileNames=valdFiles, batchSize=5000, maxItemsInQueue=5000, shuffle=False)
    testDataSet  = DataSetBIN(binFileNames=testFiles, batchSize=5000, maxItemsInQueue=5000, shuffle=False)
    modelTrainer.DefineNewParams(saveName,
                                trainDataSet,
                                validationDataSet,
                                testDataSet)
    trainingPL = TrainingPlaceholder()
    trainUpdateOp, trainLossOp, valdLossOp = GetSliceCNN(
            trainingDataSet,
            validationDataSet,
            testDataSet,
            trainingPL,
            learningRateName='LEARNING_RATE',
            keepProbName='KEEP_PROB',
            optionalHiddenLayerUnits=0,
            downscaleRate=None)
    with tf.Session() as sess:
        modelTrainer.TrainModel(self, sess, trainUpdateOp, trainLossOp, valdLossOp)

if __name__ == '__main__':
    xTrainFile = get('DATA.SLICES.X_SLICES_TRAIN')
    xValdFile  = get('DATA.SLICES.X_SLICES_VALD')
    xTestFile  = get('DATA.SLICES.X_SLICES_TEST')

    yTrainFile = get('DATA.SLICES.Y_SLICES_TRAIN')
    yValdFile  = get('DATA.SLICES.Y_SLICES_VALD')
    yTestFile  = get('DATA.SLICES.Y_SLICES_TEST')

    zTrainFile = get('DATA.SLICES.Z_SLICES_TRAIN')
    zValdFile  = get('DATA.SLICES.Z_SLICES_VALD')
    zTestFile  = get('DATA.SLICES.Z_SLICES_TEST')

    modelTrainer = ModelTrainerBIN()
    RunTestOnDirs(modelTrainer, [xTrainFile], [xValdFile], [xTestFile]])
