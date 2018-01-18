import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from data_scripts.DataPlotter import PlotTrainingValidationLoss, PlotComparisonBarChart
from data_scripts.DataSetNPY import DataSetNPY
from sklearn.model_selection import train_test_split, KFold
from utils import saveModel
from utils.config import get
from placeholders.shared_placeholders import *
from datetime import datetime

class ModelTrainerNPY(object):
    def __init__(self, summaryDir=get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'), checkpointDir=get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR')):
        self.summaryDir = summaryDir
        self.checkpointDir = checkpointDir
        self.dateString = datetime.now().strftime('%I:%M%p_%B_%d_%Y')

    def DefineFeedDict(batchArrays, batchLabels, matricesPL, labelsPL, trainingPL, isTraining=False):
        """
        Defines a tensorflow feed dict for running operations
        """
        feed_dict = {
            matricesPL: batchArrays,
            labelsPL: batchLabels,
            trainingPL: isTraining
        }
        return feed_dict

    def GetEvaluatedLoss(sess, dataSet, lossFunction, matricesPL, labelsPL, trainingPL):
        """
        Returns the evaluated loss of the current model on the given loss function
        """
        accumulatedLoss = 0
        for i in range(dataSet.numExamples):
            batchArray, batchLabel = dataSet.NextBatch(batchSize=1, shuffle=False)
            feed_dict = DefineFeedDict(batchArray, batchLabel, matricesPL, labelsPL, trainingPL)
            accumulatedLoss += sess.run(lossFunction, feed_dict=feed_dict)
        accumulatedLoss = accumulatedLoss / dataSet.numExamples

        return accumulatedLoss

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
                   numberOfSteps, batchSize, trainSummaryWriter, validationSummaryWriter):
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

        ############# DEFINE SUMMARY PLACEHOLDERS ##############
        trainLossPlaceholder = tf.placeholder(tf.float32, shape=(), name='trainLossPlaceholder')
        validationLossPlaceholder = tf.placeholder(tf.float32, shape=(), name='validationLossPlaceholder')
        trainSummary = tf.summary.scalar('trainingLoss', trainLossPlaceholder)
        validationSummary = tf.summary.scalar('validationLoss', validationLossPlaceholder)

        for batch_index in range(numberOfSteps):
            ############# RUN TRAINING OPERATIONS #############
            batchArrays, batchLabels = splitTrainSet.NextBatch(batchSize)
            feed_dict = DefineFeedDict(batchArrays, batchLabels, matricesPL, labelsPL, trainingPL, isTraining=True)
            sess.run([trainOperation, extraUpdateOps], feed_dict=feed_dict)

            ############# REPORT TRAINING PROGRESS #############
            trainingLoss, validationLoss, shouldUse = ReportProgress(sess, batch_index, lossFunction, matricesPL, labelsPL, splitTrainSet, splitValidationSet, trainingPL)
            if shouldUse:
                accumulatedTrainingLoss.append(trainingLoss)
                accumulatedValidationLoss.append(validationLoss)

                trainSummaryWriter.add_summary(sess.run(trainSummary, feed_dict={trainLossPlaceholder: trainingLoss}), batch_index)
                validationSummaryWriter.add_summary(sess.run(validationSummary, feed_dict={validationLossPlaceholder: validationLoss}), batch_index)

            ############# SAVE TRAINED MODEL #############
            SaveModel(sess, batch_index, saver, savePath)

        return (accumulatedTrainingLoss, accumulatedValidationLoss)

    def CrossValidateModelParameters(splitTrainSet, matricesPL, labelsPL, trainingPL, predictionLayer, trainOperation, lossFunction, savePath, saveName,
                                     numberOfSteps, batchSize d):
        """
        Trains a model using 5-fold cross validation on the given data set.
        Puts a plot of the results in the ../plots/ directory, and returns
        the average final validation performance.
        """
        ########## DEFINE DATA ##########
        dataDirectory = splitTrainSet.numpyDirectory
        X = splitTrainSet.numpyFileList
        Y = splitTrainSet.labels
        folder = KFold(n_splits=5, shuffle=False)
        accumulatedTrainingLoss = []
        accumulatedValidationLoss = []
        splitIndex = 0

        with tf.Session() as sess:
            for tIndex, vIndex in folder.split(X):
                splitIndex += 1
                print('-------------Split: %i-------------' % splitIndex)

                ########## CREATE SUMMARY WRITER ##########
                trainSummarydir = '{}{}/{}/train/split_{}'.format(self.summaryDir, self.dateString, saveName, splitIndex)
                trainSummaryWriter = tf.summary.FileWriter(trainSummarydir)
                validationSummaryDir = '{}{}/{}/validation/split_{}'.format(self.summaryDir, self.dateString, saveName, splitIndex)
                validationSummaryWriter = tf.summary.FileWriter(validationSummaryDir)

                ########## INITIALIZE VARIABLES ##########
                sess.run(tf.global_variables_initializer())

                ########## DEFINE THE DATA SET ##########
                fileSavePath = savePath + '_split%i.ckpt' % splitIndex
                splitTrainSet = DataSetNPY(numpyDirectory=dataDirectory, numpyFileList=X[tIndex], labels=Y[tIndex])
                splitValidationSet = DataSetNPY(numpyDirectory=dataDirectory, numpyFileList=X[vIndex], labels=Y[vIndex])

                ########## TRAIN THE MODEL ##########
                foldTrainingLosses, foldValidationLosses = TrainModel(sess, splitTrainSet, splitValidationSet,
                                                            matricesPL, labelsPL, trainingPL, predictionLayer, trainOperation,
                                                            lossFunction, fileSavePath, numberOfSteps, batchSize, trainSummaryWriter, validationSummaryWriter)
                accumulatedTrainingLoss.append(foldTrainingLosses)
                accumulatedValidationLoss.append(foldValidationLosses)

                ########## CLOSE THE SUMMARY WRITER ##########
                trainSummaryWriter.close()
                testSummaryWriter.close()

        ########## PLOT THE RESULTS OF CROSS VALIDATION ##########
        accumulatedTrainingLoss = np.array(accumulatedTrainingLoss)
        accumulatedValidationLoss = np.array(accumulatedValidationLoss)
        PlotTrainingValidationLoss(accumulatedTrainingLoss, accumulatedValidationLoss, saveName, 'plots/{}/{}.png'.format(self.dateString, saveName))

        ########## GET AVERAGE VALIDATION PERFORMANCE ##########
        averageFinalValidationPerformance = np.mean(accumulatedValidationLoss[:, -1])
        return averageFinalValidationPerformance

    def RunCrossValidation(dataSet, matricesPL, labelsPL, predictionLayers, trainOperations,
                                     lossFunctions, trainingPL, numberOfStepsArray, batchSizes, saveNames):
        if not os.path.exists('plots/{}'.format(self.dateString)):
            os.makedirs('plots/{}'.format(self.dateString))
        if not os.path.exists('{}{}/'.format(self.checkpointDir, self.dateString)):
            os.makedirs('{}{}/'.format(self.checkpointDir, self.dateString))

        ########## WRITE THE GRAPH TO THE SUMMARY FILE ##########
        summaryDir = '{}{}/'.format(self.summaryDir, self.dateString)
        graphWriter = tf.summary.FileWriter(summaryDir, graph=tf.get_default_graph())
        graphWriter.close()

        ########## SPLIT DATA INTO TRAIN AND TEST ##########
        X_train, X_test, y_train, y_test = train_test_split(dataSet.numpyFileList, dataSet.labels, test_size=0.1)
        splitTrainSet = DataSetNPY(dataSet.numpyDirectory, X_train, y_train)
        splitTestSet = DataSetNPY(dataSet.numpyDirectory, X_test, y_test)

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
            savePath = '{}{}/{}'.format(self.checkpointDir, self.dateString, saveName)

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
                    fileSavePath = savePath = '{}{}/{}_split1.ckpt'.format(self.checkpointDir, self.dateString, saveName)
                    print(fileSavePath)
                    saver = saveModel.restore(sess, fileSavePath)
                    testLoss = GetEvaluatedLoss(sess, splitTestSet, lossFunction, matricesPL, labelsPL, trainingPL)
                    print('Best model had test loss: %f' % testLoss)
            index += 1
        savePath = 'plots/{}/modelComparison.png'.format(self.dateString)
        PlotComparisonBarChart(performances=finalValidationPerformances, names=saveNames, savePath=savePath)
