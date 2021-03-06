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
    def __init__(self,
                 summaryDir=get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'),
                 checkpointDir=get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'),
                 preloadValidationSets=False):
        self.preloadValidationSets = preloadValidationSets
        self.summaryDir = summaryDir
        self.checkpointDir = checkpointDir
        self.dateString = datetime.now().strftime('%I:%M%p_%B_%d_%Y')

        ############# DEFINE SUMMARY PLACEHOLDERS ##############
        self.trainLossPlaceholder = tf.placeholder(tf.float32, shape=(), name='trainLossPlaceholder')
        self.validationLossPlaceholder = tf.placeholder(tf.float32, shape=(), name='validationLossPlaceholder')
        self.trainSummary = tf.summary.scalar('trainingLoss', self.trainLossPlaceholder)
        self.validationSummary = tf.summary.scalar('validationLoss', self.validationLossPlaceholder)

    def DefineFeedDict(self, batchArrays, batchLabels, matricesPL, labelsPL, trainingPL, isTraining=False):
        """
        Defines a tensorflow feed dict for running operations
        """
        feed_dict = {
            matricesPL: batchArrays,
            labelsPL: batchLabels,
            trainingPL: isTraining
        }
        return feed_dict

    def GetEvaluatedLoss(self, sess, dataSet, lossFunction, matricesPL, labelsPL, trainingPL):
        """
        Returns the evaluated loss of the current model on the given loss function
        """
        accumulatedLoss = 0
        if dataSet.isPreloaded:
            batchArray, batchLabel = dataSet.GetPreloadedData()
            feed_dict = self.DefineFeedDict(batchArray, batchLabel, matricesPL, labelsPL, trainingPL)
            accumulatedLoss += sess.run(lossFunction, feed_dict=feed_dict)

        else:
            for i in range(dataSet.numExamples):
                batchArray, batchLabel = dataSet.NextBatch(batchSize=1, shuffle=False)
                feed_dict = self.DefineFeedDict(batchArray, batchLabel, matricesPL, labelsPL, trainingPL)
                accumulatedLoss += sess.run(lossFunction, feed_dict=feed_dict)
            accumulatedLoss = accumulatedLoss / dataSet.numExamples

        return accumulatedLoss

    def ReportProgress(self, sess, step, lossFunction, matricesPL, labelsPL, splitTrainSet, splitValdSet, trainingPL, stepSize=50):
        """
        Reports training and validation loss every stepSize steps
        """
        if step % stepSize == 0:
            trainingLoss = self.GetEvaluatedLoss(sess, splitTrainSet, lossFunction, matricesPL, labelsPL, trainingPL)
            validationLoss = self.GetEvaluatedLoss(sess, splitValdSet, lossFunction, matricesPL, labelsPL, trainingPL)
            print('Step: %d, Evaluated Training Loss: %f, Evaluated Validation Loss: %f' % (step, trainingLoss, validationLoss))
            return (trainingLoss, validationLoss, True)
        else:
            return (None, None, False)

    def SaveModel(self, sess, step, saver, path, stepSize=50):
        """
        Saves the model to path every stepSize steps
        """
        if step % stepSize == 0:
            saver.save(sess, path)
            print('Step: %d, Saved model to path  %s' % (step, path))

    def TrainModel(self, sess, splitTrainSet, splitValidationSet, matricesPL, labelsPL, trainingPL, predictionLayer, trainOperation, lossFunction, savePath,
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
        bestValidationLoss = math.inf
        bestLossStepIndex = 0

        for batch_index in range(numberOfSteps):
            ############# RUN TRAINING OPERATIONS #############
            batchArrays, batchLabels = splitTrainSet.NextBatch(batchSize)
            feed_dict = self.DefineFeedDict(batchArrays, batchLabels, matricesPL, labelsPL, trainingPL, isTraining=True)
            sess.run([trainOperation, extraUpdateOps], feed_dict=feed_dict)

            ############# REPORT TRAINING PROGRESS #############
            trainingLoss, validationLoss, shouldUse = self.ReportProgress(sess, batch_index, lossFunction, matricesPL, labelsPL, splitTrainSet, splitValidationSet, trainingPL)
            if shouldUse:
                accumulatedTrainingLoss.append(trainingLoss)
                accumulatedValidationLoss.append(validationLoss)

                trainSummaryWriter.add_summary(sess.run(self.trainSummary, feed_dict={self.trainLossPlaceholder: trainingLoss}), batch_index)
                validationSummaryWriter.add_summary(sess.run(self.validationSummary, feed_dict={self.validationLossPlaceholder: validationLoss}), batch_index)

                ############# SAVE TRAINED MODEL #############
                if validationLoss < bestValidationLoss:
                    bestLossStepIndex = batch_index
                    bestValidationLoss = validationLoss
                    self.SaveModel(sess, batch_index, saver, savePath)

        print('Step: {}, best validation loss: {}'.format(bestLossStepIndex, bestValidationLoss))
        return (accumulatedTrainingLoss, accumulatedValidationLoss, bestValidationLoss)

    def CrossValidateModelParameters(self, splitTrainSet, matricesPL, labelsPL, trainingPL, predictionLayer, trainOperation, lossFunction, savePath, saveName,
                                     numberOfSteps, batchSize):
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
        bestValidationLosses = []
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
                splitTrainSet = DataSetNPY(numpyDirectory=dataDirectory, numpyFileList=X[tIndex], labels=Y[tIndex], reshapeBatches=splitTrainSet.reshapeBatches)
                splitValidationSet = DataSetNPY(numpyDirectory=dataDirectory, numpyFileList=X[vIndex], labels=Y[vIndex], reshapeBatches=splitTrainSet.reshapeBatches)
                if self.preloadValidationSets:
                    splitValidationSet.PreloadData()

                ########## TRAIN THE MODEL ##########
                foldTrainingLosses, foldValidationLosses, bestValidationLoss = self.TrainModel(sess, splitTrainSet, splitValidationSet,
                                                            matricesPL, labelsPL, trainingPL, predictionLayer, trainOperation,
                                                            lossFunction, fileSavePath, numberOfSteps, batchSize, trainSummaryWriter, validationSummaryWriter)
                accumulatedTrainingLoss.append(foldTrainingLosses)
                accumulatedValidationLoss.append(foldValidationLosses)
                bestValidationLosses.append(bestValidationLoss)

                ########## CLOSE THE SUMMARY WRITER ##########
                trainSummaryWriter.close()
                validationSummaryWriter.close()

        ########## PLOT THE RESULTS OF CROSS VALIDATION ##########
        accumulatedTrainingLoss = np.array(accumulatedTrainingLoss)
        accumulatedValidationLoss = np.array(accumulatedValidationLoss)
        PlotTrainingValidationLoss(accumulatedTrainingLoss, accumulatedValidationLoss, saveName, 'plots/{}/{}.png'.format(self.dateString, saveName))

        ########## GET BEST VALIDATION PERFORMANCE ##########
        averageBestValidationPerformance = np.mean(bestValidationLosses)
        return averageBestValidationPerformance

    def RunCrossValidation(self, dataSet, matricesPL, labelsPL, predictionLayers, trainOperations,
                                     lossFunctions, trainingPL, numberOfStepsArray, batchSizes, saveNames):
        if not os.path.exists('plots/{}'.format(self.dateString)):
            os.makedirs('plots/{}'.format(self.dateString))
        if not os.path.exists('{}{}/'.format(self.checkpointDir, self.dateString)):
            os.makedirs('{}{}/'.format(self.checkpointDir, self.dateString))

        ########## WRITE THE GRAPH TO THE SUMMARY FILE ##########
        summaryDir = '{}{}/'.format(self.summaryDir, self.dateString)
        graphWriter = tf.summary.FileWriter(summaryDir, graph=tf.get_default_graph())
        graphWriter.close()

        if not isinstance(dataSet, list):
            ########## SPLIT DATA INTO TRAIN AND TEST ##########
            X_train, X_test, y_train, y_test = train_test_split(dataSet.numpyFileList, dataSet.labels, test_size=0.1)
            splitTrainSet = DataSetNPY(dataSet.numpyDirectory, X_train, y_train, reshapeBatches=dataSet.reshapeBatches)
            splitTestSet = DataSetNPY(dataSet.numpyDirectory, X_test, y_test, reshapeBatches=dataSet.reshapeBatches)

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

            if isinstance(dataSet, list):
                ########## SPLIT DATA INTO TRAIN AND TEST ##########
                X_train, X_test, y_train, y_test = train_test_split(dataSet[index].numpyFileList, dataSet[index].labels, test_size=0.1)
                splitTrainSet = DataSetNPY(dataSet[index].numpyDirectory, X_train, y_train, reshapeBatches=dataSet[index].reshapeBatches)
                splitTestSet = DataSetNPY(dataSet[index].numpyDirectory, X_test, y_test, reshapeBatches=dataSet[index].reshapeBatches)

            print('===================%s===================' % saveName)
            savePath = '{}{}/{}'.format(self.checkpointDir, self.dateString, saveName)

            ########## GET CROSS VALIDATION PERFORMANCE OF MODEL ##########
            averageBestValidationPerformance = self.CrossValidateModelParameters(splitTrainSet,
                                                    matricesPL, labelsPL, trainingPL, predictionLayer, trainOperation,
                                                    lossFunction, savePath, saveName,
                                                    numberOfSteps, batchSize)
            finalValidationPerformances.append(averageBestValidationPerformance)

            ########## DETERMINE BEST MODEL SO FAR ##########
            if (averageBestValidationPerformance < lowestLoss):
                lowestLoss = averageBestValidationPerformance
                bestIndex = index
            index += 1

        ########## PRINT CROSS VALIDATION RESULTS ##########
        print('===================CROSS VALIDATION RESULTS===================')
        for i in range(index):
            saveName = saveNames[i]
            print('Model %s had validation performance: %f' % (saveName, finalValidationPerformances[i]))
        print('===================BEST MODEL===================')
        print('Best model was %s with validation performance of %f' % (saveNames[bestIndex], finalValidationPerformances[bestIndex]))

        predictionLayer = predictionLayers[bestIndex]
        lossFunction = lossFunctions[bestIndex]
        trainOperation = trainOperations[bestIndex]
        numberOfSteps = numberOfStepsArray[bestIndex]
        batchSize = batchSizes[bestIndex]
        saveName = saveNames[bestIndex]
        if isinstance(dataSet, list):
            ########## SPLIT DATA INTO TRAIN AND TEST ##########
            X_train, X_test, y_train, y_test = train_test_split(dataSet[bestIndex].numpyFileList, dataSet[bestIndex].labels, test_size=0.1)
            splitTrainSet = DataSetNPY(dataSet[bestIndex].numpyDirectory, X_train, y_train, reshapeBatches=dataSet[bestIndex].reshapeBatches)
            splitTestSet = DataSetNPY(dataSet[bestIndex].numpyDirectory, X_test, y_test, reshapeBatches=dataSet[bestIndex].reshapeBatches)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            fileSavePath = savePath = '{}{}/{}_split1.ckpt'.format(self.checkpointDir, self.dateString, saveName)
            print(fileSavePath)
            saver = saveModel.restore(sess, fileSavePath)
            testLoss = self.GetEvaluatedLoss(sess, splitTestSet, lossFunction, matricesPL, labelsPL, trainingPL)
            print('Best model had test loss: %f' % testLoss)

        savePath = 'plots/{}/modelComparison.png'.format(self.dateString)
        PlotComparisonBarChart(performances=finalValidationPerformances, names=saveNames, savePath=savePath)
