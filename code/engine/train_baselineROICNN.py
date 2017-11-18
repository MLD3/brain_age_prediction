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

if __name__ == '__main__':
    dataHolder = DataHolder(readCSVData(get('DATA.PHENOTYPICS.PATH')))
    dataHolder.getMatricesFromPath(get('DATA.MATRICES.PATH'))
    dataHolder.matricesToImages()
    dataSet = dataHolder.returnDataSet()

    ############ DEFINE PLACEHOLDERS, OPERATIONS ############
    trainingPL = TrainingPlaceholder()
    matricesPL, labelsPL = MatrixPlaceholders()
    predictionLayer = baselineROICNN(matricesPL, trainingPL)
    lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)
    trainOperation = AdamOptimizer(lossFunction, get('TRAIN.ROI_BASELINE.LEARNING_RATE'))
    stepCount = get('TRAIN.ROI_BASELINE.NB_STEPS')
    batchSize = get('TRAIN.ROI_BASELINE.BATCH_SIZE')

    saveNames = CreateNameArray(['MatrixTest'], ['MatrixLabel'], ['roiBaseline'], ['ADAM'], ['100'], ['32', '128'])
    RunCrossValidation(dataSet, [matricesPL], [labelsPL], [predictionLayer], [trainOperation],
                                     lossFunction, trainingPL, [stepCount], [batchSize, 128], saveNames)
