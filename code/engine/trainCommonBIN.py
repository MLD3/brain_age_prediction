import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
from data_scripts.DataSetBIN import DataSetBIN
from sklearn.model_selection import train_test_split, KFold
from utils import saveModel
from utils.config import get
from placeholders.shared_placeholders import *
from datetime import datetime

class ModelTrainerBIN(object):
    def __init__(self):
        self.dateString = datetime.now().strftime('%I:%M%p_%B_%d_%Y')

    def DefineNewParams(self,
                        saveName,
                        trainDataSet,
                        validationDataSet,
                        testDataSet,
                        summaryDir=get('TRAIN.CNN_BASELINE.SUMMARIES_DIR'),
                        checkpointDir=get('TRAIN.CNN_BASELINE.CHECKPOINT_DIR'),
                        numberOfSteps=get('TRAIN.DEFAULTS.TEST_NB_STEPS'),
                        batchStepsBetweenSummary=200
                        ):
        self.saveName = saveName
        self.trainDataSet = trainDataSet
        self.validationDataSet = validationDataSet
        self.testDataSet = testDataSet

        if not os.path.exists('{}{}'.format(checkpointDir, self.dateString)):
            os.makedirs('{}{}'.format(checkpointDir, self.dateString))
        self.checkpointDir = "{}{}/{}".format(checkpointDir, self.dateString, self.saveName)
        self.numberOfSteps = numberOfSteps
        self.batchStepsBetweenSummary = batchStepsBetweenSummary

        summaryDir = '{}{}/{}/'.format(summaryDir, self.dateString, self.saveName)
        self.writer = tf.summary.FileWriter(summaryDir, graph=tf.get_default_graph())

        self.trainLossPlaceholder = tf.placeholder(tf.float32, shape=(), name='trainLossPlaceholder')
        self.validationLossPlaceholder = tf.placeholder(tf.float32, shape=(), name='validationLossPlaceholder')
        self.trainSummary = tf.summary.scalar('trainingLoss', self.trainLossPlaceholder)
        self.validationSummary = tf.summary.scalar('validationLoss', self.validationLossPlaceholder)

    def GetPerformanceThroughSet(self, sess, lossOp, numberIters=50):
        accumulatedLoss = 0
        for i in range(numberIters):
            currentLoss = sess.run(lossOp)
            accumulatedLoss += currentLoss
        accumulatedLoss = accumulatedLoss / numberIters
        return accumulatedLoss

    def GetBootstrapTestPerformance(self, sess, trainingPL, testLossOp, bootstrapLossOp):
        numReps = 1000
        confidenceInterval = 0.95
        alpha = (1.0 - confidenceInterval) * 0.5
        lowerBoundIndex = numReps * alpha
        upperBoundIndex = numReps * (1.0 - alpha)

        bootstrapPerformances = np.zeros(numReps)
        for i in range(numReps):
            print('Getting bootstrap test performance, iteration {} out of {}'.format(i, numReps), end='\r')

            bootstrapPerformances[i] = self.GetPerformanceThroughSet(sess, bootstrapLossOp)
        bootstrapPerformances = np.sort(bootstrapPerformances)

        bootstrapPlaceholder = tf.placeholder(dtype=tf.float32, shape=(numReps,), name='bootstrapPlaceholder')
        bootstrapHist =  tf.summary.histogram('Bootstrap Test Performance', bootstrapPlaceholder)
        pointPerformance = self.GetPerformanceThroughSet(sess, testLossOp)

        histSummary = \
            sess.run(bootstrapHist,
                     feed_dict={
                        bootstrapPlaceholder: bootstrapPerformances,
                        trainingPL: False})

        self.writer.add_summary(histSummary, 1)

        return pointPerformance, bootstrapPerformances[lowerBoundIndex], bootstrapPerformances[upperBoundIndex]

    def SaveModel(self, sess, step, saver):
        """
        Saves the model to path every stepSize steps
        """
        saver.save(sess, self.checkpointDir)
        print('STEP {}: saved model to path {}'.format(step, self.checkpointDir))

    def TrainModel(self, sess, trainingPL, trainUpdateOp, trainLossOp, valdLossOp, testLossOp, bootstrapLossOp):
        # Start the threads to read in data
        print('Starting queue runners...')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Initialize relevant variables
        print('Initializing variables')
        sess.run(tf.global_variables_initializer())

        # Collect summary and graph update operations
        extraUpdateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Restore a model if it exists in the indicated directory
        saver = saveModel.restore(sess, self.checkpointDir)

        bestValidationLoss = math.inf
        bestLossStepIndex = 0

        print('Beginning Training...')
        for batchIndex in range(self.numberOfSteps):
            sess.run([trainUpdateOp, extraUpdateOps], feed_dict={
                trainingPL: True
                })

            if batchIndex % self.batchStepsBetweenSummary == 0:
                trainingLoss = \
                    sess.run(trainLossOp, feed_dict={
                    trainingPL: False
                    })
                validationLoss = self.GetPerformanceThroughSet(sess, valdLossOp)

                print('STEP {}: Training Loss = {}, Validation Loss = {}'.format(
                            batchIndex,
                            trainingLoss,
                            validationLoss))
                self.writer.add_summary(
                    sess.run(
                        self.trainSummary,
                        feed_dict={
                            self.trainLossPlaceholder: trainingLoss
                        }),
                    batchIndex)
                self.writer.add_summary(
                    sess.run(
                        self.validationSummary,
                        feed_dict={
                            self.validationLossPlaceholder: validationLoss
                        }),
                    batchIndex)

                if validationLoss < bestValidationLoss:
                    bestLossStepIndex = batchIndex
                    bestValidationLoss = validationLoss
                    self.SaveModel(sess, batchIndex, saver)

        pointTestPerformance, lowerBound, upperBound = \
            self.GetBootstrapTestPerformance(sess=sess,
                                             trainingPL=trainingPL,
                                             testLossOp=testLossOp,
                                             bootstrapLossOp=bootstrapLossOp)
        self.writer.close()
        coord.request_stop()
        coord.join(threads)
        print("STEP {}: Best Validation Loss = {}".format(bestLossStepIndex, bestValidationLoss))
        print("Model {} had test performance: {} ({}, {})".format(
            self.saveName,
            pointTestPerformance,
            lowerBound,
            upperBound))
