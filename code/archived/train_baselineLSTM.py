import tensorflow as tf
import numpy as np
import pandas as pd
import math
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from data_scripts.DataPlotter import PlotTrainingValidationLoss
from data_scripts.DataSetNPY import DataSetNPY
from sklearn.model_selection import train_test_split, KFold
from model.build_baselineLSTM import baselineLSTM
from utils import saveModel
from utils.config import get
from placeholders.shared_placeholders import *
from itertools import product
from engine.trainCommonNPY import *


def GetBaselineLSTMModel(timecoursePL, learningRateName='LEARNING_RATE', stepCountName='NB_STEPS',
                        batchSizeName='BATCH_SIZE', optimizer='ADAM'):
    ############ DEFINE PLACEHOLDERS, LOSS ############
    predictionLayer = baselineLSTM(timecoursePL=timecoursePL)
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)

    ############ DEFINE OPTIMIZER ############
    if optimizer == 'ADAM':
        trainOperation = AdamOptimizer(lossFunction, get('TRAIN.LSTM_BASELINE.%s' % learningRateName), clipGrads=True)
    elif optimizer == 'GRAD_DECAY':
        trainOperation = ScheduledGradOptimizer(lossFunction, baseLearningRate=get('TRAIN.LSTM_BASELINE.%s' % learningRateName))

    ############ DEFINE LEARNING PARAMETERS ############
    stepCount = get('TRAIN.LSTM_BASELINE.%s' % stepCountName)
    batchSize = get('TRAIN.LSTM_BASELINE.%s' % batchSizeName)

    return predictionLayer, lossFunction, trainOperation, stepCount, batchSize

if __name__ == '__main__':
    PhenotypicsDF = readCSVData(get('DATA.PHENOTYPICS.PATH'))
    StructuralDataDir = get('DATA.TIMECOURSES.NUMPY_PATH')
    fileList = np.array([str(subject) + '.npy' for subject in PhenotypicsDF['Subject'].tolist()])
    labels = np.array(PhenotypicsDF['AgeYears'].tolist())
    dataSet = DataSetNPY(numpyDirectory=StructuralDataDir, numpyFileList=fileList, labels=labels, reshapeBatches=False)

    trainingPL = TrainingPlaceholder()
    timecoursePL, labelsPL = TimecoursePlaceholders()
    predictionLayers = []
    trainOperations = []
    lossFunctions = []
    stepCountArray = []
    batchSizeArray = []
    saveNames = []

    with tf.variable_scope('LSTMStandard'):
        predictionLayer, lossFunction, trainOperation, stepCount, batchSize = GetBaselineLSTMModel(timecoursePL)
        predictionLayers.append(predictionLayer)
        trainOperations.append(trainOperation)
        lossFunctions.append(lossFunction)
        stepCountArray.append(stepCount)
        batchSizeArray.append(batchSize)
        saveNames.append(tf.contrib.framework.get_name_scope())

    trainer = ModelTrainerNPY(summaryDir=get('TRAIN.LSTM_BASELINE.SUMMARIES_DIR'), checkpointDir=get('TRAIN.LSTM_BASELINE.CHECKPOINT_DIR'))
    trainer.RunCrossValidation(dataSet, timecoursePL, labelsPL, predictionLayers, trainOperations,
                                     lossFunctions, trainingPL, stepCountArray, batchSizeArray, saveNames)
