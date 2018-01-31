import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
from sklearn.model_selection import train_test_split, KFold
from utils import saveModel
from utils.config import get
from placeholders.shared_placeholders import *
from datetime import datetime

class ModelTrainer(object):
    def __init__(self):
        self.dateString = datetime.now().strftime('%I:%M%p_%B_%d_%Y')

    def DefineNewParams(self,
                        trainDataSet,
                        validationDataSet,
                        testDataSet,
                        summaryDir,
                        checkpointDir,
                        numberOfSteps=get('TRAIN.DEFAULTS.TEST_NB_STEPS'),
                        batchStepsBetweenSummary=500
                        ):
        self.trainDataSet = trainDataSet
        self.validationDataSet = validationDataSet
        self.testDataSet = testDataSet

        if not os.path.exists(checkpointDir):
            os.makedirs(checkpointDir)
        self.checkpointDir = checkpointDir
        self.numberOfSteps = numberOfSteps
        self.batchStepsBetweenSummary = batchStepsBetweenSummary

        self.writer = tf.summary.FileWriter(summaryDir)

        self.trainLossPlaceholder = tf.placeholder(tf.float32, shape=(), name='trainLossPlaceholder')
        self.validationLossPlaceholder = tf.placeholder(tf.float32, shape=(), name='validationLossPlaceholder')
        self.trainSummary = tf.summary.scalar('trainingLoss', self.trainLossPlaceholder)
        self.validationSummary = tf.summary.scalar('validationLoss', self.validationLossPlaceholder)

    def GetPerformanceThroughSet(self, sess, lossOp, numberIters=75):
        accumulatedLoss = 0
        for i in range(numberIters):
            currentLoss = sess.run(lossOp)
            accumulatedLoss += currentLoss
        accumulatedLoss = accumulatedLoss / numberIters
        return accumulatedLoss

    def SaveModel(self, sess, step, saver):
        """
        Saves the model to path every stepSize steps
        """
        saver.save(sess, self.checkpointDir)
        print('STEP {}: saved model to path {}'.format(step, self.checkpointDir))

    def TrainModel(self, sess, trainingPL, trainUpdateOp, trainLossOp, valdLossOp, testLossOp):
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
            trainingLoss, _, _ = sess.run([trainLossOp, trainUpdateOp, extraUpdateOps], feed_dict={
                trainingPL: True
                })

            if batchIndex % self.batchStepsBetweenSummary == 0:
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

        testLoss = self.GetPerformanceThroughSet(sess, testLossOp)
        self.writer.close()
        coord.request_stop()
        coord.join(threads)
        print("STEP {}: Best Validation Loss = {}".format(bestLossStepIndex, bestValidationLoss))
        print("Model had test performance: {}".format(testLoss))
