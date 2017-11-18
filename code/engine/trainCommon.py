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

def DefineFeedDict(dataSet, matricesPL, labelsPL, trainingPL, isTraining=False):
    """
    Defines a tensorflow feed dict for running operations
    """
    feed_dict = {
        matricesPL: dataSet.images,
        labelsPL: dataSet.labels,
        trainingPL: isTraining
    }
    return feed_dict

def GetEvaluatedLoss(sess, dataSet, lossFunction, matricesPL, labelsPL, trainingPL):
    """
    Returns the evaluated loss of the current model on the given loss function
    """
    feed_dict = DefineFeedDict(dataSet, matricesPL, labelsPL, trainingPL)
    return sess.run(lossFunction, feed_dict=feed_dict)

def ReportProgress(sess, step, lossFunction, matricesPL, labelsPL, splitTrainSet, splitValdSet, trainingPL, stepSize=50):
    """
    Reports training and validation loss every stepSize steps
    """
    if step % stepSize == 0:
        trainingLoss = GetEvaluatedLoss(sess, splitTrainSet, lossFunction, matricesPL, labelsPL, trainingPL)
        validationLoss = GetEvaluatedLoss(sess, splitValdSet, lossFunction, matricesPL, labelsPL, trainingPL)
        print('Step: %d, Evaluated Training Loss: %f, Evaluated Validation Loss: %f' % (step, trainingLoss, validationLoss))
        return (trainingLoss, validationLoss, True)
    else:
        return (None, None, False)

def SaveModel(sess, step, saver, path, stepSize=100):
    """
    Saves the model to path every stepSize steps
    """
    if step % stepSize == 0:
        saver.save(sess, path)
        print('Step: %d, Saved model to path  %s' % (step, path))

def TrainModel(sess, splitTrainSet, splitValidationSet, matricesPL, labelsPL, trainingPL, predictionLayer, trainOperation, lossFunction, savePath,
               numberOfSteps=get('TRAIN.ROI_BASELINE.NB_STEPS'), batchSize=get('TRAIN.ROI_BASELINE.BATCH_SIZE')):
    """
    Trains a model defined by matricesPL, labelsPL, predictionLayer, trainOperation and lossFunction
    over numberOfSteps steps with batch size batchSize. Uses savePath to save the model.
    """
    extraUpdateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    ############# Define tf saver #############
    saver = saveModel.restore(sess, savePath)

    ############# DEFINE ARRAYS TO HOLD LOSS #############
    accumulatedTrainingLoss = []
    accumulatedValidationLoss = []

    for batch_index in range(numberOfSteps):
        ############# RUN TRAINING OPERATIONS #############
        batch_images, batch_labels = splitTrainSet.next_batch(batchSize)
        feed_dict = DefineFeedDict(DataSet(batch_images, batch_labels), matricesPL, labelsPL, trainingPL, isTraining=True)
        sess.run([trainOperation, extraUpdateOps], feed_dict=feed_dict)

        ############# REPORT TRAINING PROGRESS #############
        trainingLoss, validationLoss, shouldUse = ReportProgress(sess, batch_index, lossFunction, matricesPL, labelsPL, splitTrainSet, splitValidationSet, trainingPL)
        if shouldUse:
            accumulatedTrainingLoss.append(trainingLoss)
            accumulatedValidationLoss.append(validationLoss)

        ############# SAVE TRAINED MODEL #############
        SaveModel(sess, batch_index, saver, savePath)

    return (accumulatedTrainingLoss, accumulatedValidationLoss)

def CrossValidateModelParameters(splitTrainSet, matricesPL, labelsPL, trainingPL, predictionLayer, trainOperation, lossFunction, savePath, saveName,
                                 numberOfSteps=get('TRAIN.ROI_BASELINE.NB_STEPS'), batchSize=get('TRAIN.ROI_BASELINE.BATCH_SIZE')):
    """
    Trains a model using 5-fold cross validation on the given data set.
    Puts a plot of the results in the ../plots/ directory, and returns
    the average final validation performance.
    """
    ########## DEFINE DATA ##########
    X = splitTrainSet.images
    Y = splitTrainSet.labels
    folder = KFold(n_splits=5, shuffle=False)
    accumulatedTrainingLoss = []
    accumulatedValidationLoss = []
    splitIndex = 0

    for tIndex, vIndex in folder.split(X):
        ########## TRAIN THE MODEL ##########
        splitIndex += 1
        print('-------------Split: %i-------------' % splitIndex)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            fileSavePath = savePath + '_split%i.ckpt' % splitIndex
            splitTrainSet = DataSet(images=X[tIndex], labels=Y[tIndex], numClasses=1)
            splitValidationSet = DataSet(images=X[vIndex], labels=Y[vIndex], numClasses=1)
            foldTrainingLosses, foldValidationLosses = TrainModel(sess, splitTrainSet, splitValidationSet,
                                                        matricesPL, labelsPL, trainingPL, predictionLayer, trainOperation,
                                                        lossFunction, fileSavePath, numberOfSteps, batchSize)
            accumulatedTrainingLoss.append(foldTrainingLosses)
            accumulatedValidationLoss.append(foldValidationLosses)

    ########## PLOT THE RESULTS OF CROSS VALIDATION ##########
    accumulatedTrainingLoss = np.array(accumulatedTrainingLoss)
    accumulatedValidationLoss = np.array(accumulatedValidationLoss)
    PlotTrainingValidationLoss(accumulatedTrainingLoss, accumulatedValidationLoss, saveName, 'plots/' + saveName + '.png')

    ########## GET AVERAGE VALIDATION PERFORMANCE ##########
    averageFinalValidationPerformance = np.mean(accumulatedValidationLoss[:, -1])
    return averageFinalValidationPerformance

def RunCrossValidation(dataSet, matrixPlaceholders, labelPlaceholders, predictionLayers, trainOperations,
                                 lossFunction, trainingPL, numberOfStepsArray, batchSizes, saveNames):
    ########## SPLIT DATA INTO TRAIN AND TEST ##########
    X_train, X_test, y_train, y_test = train_test_split(dataSet.images, dataSet.labels, test_size=0.2)
    splitTrainSet = DataSet(X_train, y_train)
    splitTestSet = DataSet(X_test, y_test)

    ########## ITERATE OVER ALL MODELS ##########
    index = 0
    bestIndex = -1
    lowestLoss = math.inf
    finalValidationPerformances = []
    for matricesPL, labelsPL, predictionLayer, trainOperation, numberOfSteps, batchSize in product(matrixPlaceholders, labelPlaceholders, predictionLayers, trainOperations, numberOfStepsArray, batchSizes):
        saveName = saveNames[index]
        print('===================%s===================' % saveName)
        savePath = get('TRAIN.ROI_BASELINE.CHECKPOINT_DIR') + saveName

        ########## GET CROSS VALIDATION PERFORMANCE OF MODEL ##########
        averageFinalValidationPerformance = CrossValidateModelParameters(splitTrainSet,
                                                matricesPL, labelsPL, trainingPL, predictionLayer, trainOperation,
                                                lossFunction, savePath, saveName,
                                                numberOfSteps, batchSize)
        finalValidationPerformances.append(averageFinalValidationPerformance)

        ########## DETERMINE BEST MODEL SO FAR ##########
        if (averageFinalValidationPerformance < lowestLoss):
            lowestLoss = averageFinalValidationPerformance
            bestIndex = index
        index += 1

    ########## PRINT CROSS VALIDATION RESULTS ##########
    print('===================CROSS VALIDATION RESULTS===================')
    for i in range(index):
        saveName = saveNames[i]
        print('Model %s had validation performance: %f' % (saveName, finalValidationPerformances[i]))
    print('===================BEST MODEL===================')
    print('Best model was %s with validation performance of %f' % (saveNames[bestIndex], finalValidationPerformances[bestIndex]))

    index = 0
    for matricesPL, labelsPL, predictionLayer, trainOperation, numberOfSteps, batchSize in product(matrixPlaceholders, labelPlaceholders, predictionLayers, trainOperations, numberOfStepsArray, batchSizes):
        if (index == bestIndex):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                fileSavePath = savePath + '_split1.ckpt'
                saver = saveModel.restore(sess, fileSavePath)
                testLoss = GetEvaluatedLoss(sess, splitTestSet, lossFunction, matricesPL, labelsPL, trainingPL)
                print('Best model had test loss: %f' % testLoss)
        index += 1

def CreateNameArray(inputPLNames, labelsPLNames, predictionLayerNames, trainOperationNames, stepCountNames, batchSizeNames):
    saveNames = []
    for matricesPL, labelsPL, predictionLayer, trainOperation, numberOfSteps, batchSize in product(inputPLNames, labelsPLNames, predictionLayerNames, trainOperationNames, stepCountNames, batchSizeNames):
        names = matricesPL + '_' + labelsPL + '_' +  predictionLayer + '_' + trainOperation + '_' + numberOfSteps + '_' + batchSize
        saveNames.append(names)
    return saveNames
