import tensorflow as tf
import numpy as np
import pandas as pd
import math
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from data_scripts.DataPlotter import PlotTrainingValidationLoss, PlotComparisonBarChart
from data_scripts.DataSet import DataSet
from sklearn.model_selection import train_test_split, KFold
from utils import saveModel
from utils.config import get
from placeholders.shared_placeholders import *
from datetime import datetime

def performanceCI(sess, dataSet, lossFunction, matricesPL, labelsPL, trainingPL):
    N = 1000
    X = dataSet.images
    Y = dataSet.labels
    bootstrap_performances = np.zeros(N)
    n = X.shape[0]
    indices = np.arange(n)

    for i in range(N):
        sample_indices = np.random.choice(indices, size=n, replace=True)
        sampleX = X[sample_indices]
        sampleY = Y[sample_indices]
        sampleDataset = DataSet(sampleX, sampleY)

        bootstrap_performances[i] = GetEvaluatedLoss(sess, sampleDataset, lossFunction, matricesPL, labelsPL, trainingPL)

    bootstrap_performances = np.sort(bootstrap_performances)
    point_performance = np.mean(bootstrap_performances)

    return (point_performance, bootstrap_performances[25], bootstrap_performances[975])

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
               numberOfSteps, batchSize):
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
                                 numberOfSteps, batchSize):
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
            if splitIndex == 5:
                (point, lower, upper) = performanceCI(sess, splitValidationSet, lossFunction, matricesPL, labelsPL, trainingPL)
                print("Confidence Interval Performance: %f (%f, %f)" % (point, lower, upper))

    ########## PLOT THE RESULTS OF CROSS VALIDATION ##########
    accumulatedTrainingLoss = np.array(accumulatedTrainingLoss)
    accumulatedValidationLoss = np.array(accumulatedValidationLoss)
    PlotTrainingValidationLoss(accumulatedTrainingLoss, accumulatedValidationLoss, saveName, 'plots/' + saveName + '.png')

    if numberOfSteps >= 1000:
        PlotTrainingValidationLoss(accumulatedTrainingLoss[:,-1000:], accumulatedValidationLoss[:,-1000:], saveName, 'plots/' + saveName + 'last1000.png')

    ########## GET AVERAGE VALIDATION PERFORMANCE ##########
    averageFinalValidationPerformance = np.mean(accumulatedValidationLoss[:, -1])
    return averageFinalValidationPerformance

def RunCrossValidation(dataSet, matricesPL, labelsPL, predictionLayers, trainOperations,
                                 lossFunctions, trainingPL, numberOfStepsArray, batchSizes, saveNames):
    ########## SPLIT DATA INTO TRAIN AND TEST ##########
    X_train, X_test, y_train, y_test = train_test_split(dataSet.images, dataSet.labels, test_size=0.1)
    splitTrainSet = DataSet(X_train, y_train)
    splitTestSet = DataSet(X_test, y_test)

    ########## ITERATE OVER ALL MODELS ##########
    index = 0
    bestIndex = -1
    lowestLoss = math.inf
    finalValidationPerformances = []
    for index in range(len(saveNames)):
        predictionLayer = predictionLayers[index]
        lossFunction = lossFunctions[index]
        trainOperation = trainOperations[index]
        numberOfSteps = numberOfStepsArray[index]
        batchSize = batchSizes[index]
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

    for index in range(len(saveNames)):
        predictionLayer = predictionLayers[index]
        lossFunction = lossFunctions[index]
        trainOperation = trainOperations[index]
        numberOfSteps = numberOfStepsArray[index]
        batchSize = batchSizes[index]
        saveName = saveNames[index]

        if (index == bestIndex):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                fileSavePath = get('TRAIN.ROI_BASELINE.CHECKPOINT_DIR') + saveName + '_split1.ckpt'
                print(fileSavePath)
                saver = saveModel.restore(sess, fileSavePath)
                testLoss = GetEvaluatedLoss(sess, splitTestSet, lossFunction, matricesPL, labelsPL, trainingPL)
                print('Best model had test loss: %f' % testLoss)
        index += 1
    savePath = 'plots/modelComparison%s.png' % datetime.now().strftime('%I:%M%p_%B_%d_%Y')
    PlotComparisonBarChart(performances=finalValidationPerformances, names=saveNames, savePath=savePath)
