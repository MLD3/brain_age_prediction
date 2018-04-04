import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetNPY import DataSetNPY
from model.buildCustomCNN import customCNN
from utils.saveModel import *
from utils.patches import pairwiseDiceTF
from utils.config import get
from engine.trainCommon import *
from placeholders.shared_placeholders import *
from engine.trainCustomCNN import GetTrainingOperation, GetDataSetInputs, DefineDataOpts, GetOps, GetArgs

def runRandomFlips():
    ParseArgs('Run 3D CNN over structural MRI volumes')
    GlobalOpts.scale = 3
    GlobalOpts.type = 'reverse'
    GlobalOpts.summaryName = 'alignmentComp'
    GlobalOpts.data = 'PNC'
    GlobalOpts.sliceIndex = None
    GlobalOpts.align = None
    GlobalOpts.numberTrials = 50
    GlobalOpts.padding = None
    GlobalOpts.batchSize = 4
    GlobalOpts.pheno = None
    GlobalOpts.validationDir = None
    GlobalOpts.regStrength = None
    GlobalOpts.learningRate = 0.0001
    GlobalOpts.maxNorm = None
    GlobalOpts.dropout = 0.6
    GlobalOpts.dataScale = 3
    GlobalOpts.pncDataType = 'AVG'
    GlobalOpts.listType = None
    DefineDataOpts(data=GlobalOpts.data, summaryName=GlobalOpts.summaryName)
    modelTrainer = ModelTrainer()
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    imagesPL, labelsPL = StructuralPlaceholders(GlobalOpts.imageBatchDims)
    trainingPL = TrainingPlaceholder()
    convLayers = [64, 32, 16, 8]
    fullyConnectedLayers = [256, 1]
    phenotypicsPL = None
    outputLayer = customCNN(imagesPL,
                            trainingPL,
                            GlobalOpts.scale,
                            convLayers,
                            fullyConnectedLayers,
                            keepProbability=GlobalOpts.dropout,
                            poolType=GlobalOpts.poolType,
                            sliceIndex=GlobalOpts.sliceIndex,
                            align=GlobalOpts.align,
                            padding=GlobalOpts.padding,
                            phenotypicsPL=phenotypicsPL,
                            randomFlips=True)
    with tf.variable_scope('LossOperations'):
        lossOp = tf.losses.mean_squared_error(labels=labelsPL, predictions=outputLayer)
        MSEOp, MSEUpdateOp = tf.metrics.mean_squared_error(labels=labelsPL, predictions=outputLayer)
        MAEOp, MAEUpdateOp = tf.metrics.mean_absolute_error(labels=labelsPL, predictions=outputLayer)
        updateOp, gradients = GetTrainingOperation(lossOp, GlobalOpts.learningRate)
    diceOp, diceUpdateOp = pairwiseDiceTF(imagesPL)
    printOps = PrintOps(ops=[MSEOp, MAEOp, diceOp],
        updateOps=[MSEUpdateOp, MAEUpdateOp, diceUpdateOp],
        names=['loss', 'MAE', 'PairwiseDice'],
        gradients=gradients)
    modelTrainer.DefineNewParams(GlobalOpts.summaryDir,
                                GlobalOpts.checkpointDir,
                                imagesPL,
                                trainingPL,
                                labelsPL,
                                trainDataSet,
                                valdDataSet,
                                testDataSet,
                                GlobalOpts.numSteps,
                                phenotypicsPL=phenotypicsPL)
    config  = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GlobalOpts.gpuMemory
    with tf.Session(config=config) as sess:
        modelTrainer.RepeatTrials(sess,
                              updateOp,
                              printOps,
                              name=GlobalOpts.name,
                              numIters=GlobalOpts.numberTrials)

if __name__ == '__main__':
    runRandomFlips()
