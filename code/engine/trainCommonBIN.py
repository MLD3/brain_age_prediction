import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from data_scripts.DataPlotter import PlotTrainingValidationLoss, PlotComparisonBarChart
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
                        numberOfSteps=get('TRAIN.DEFAULTS.NB_STEPS'),
                        batchStepsBetweenSummary=200
                        ):
        self.saveName = saveName
        self.trainDataSet = trainDataSet
        self.validationDataSet = validationDataSet
        self.testDataSet = testDataSet

        self.checkpointDir = "{}{}/{}".format(checkpointDir, self.dateString, self.saveName)
        self.numberOfSteps = numberOfSteps
        self.batchStepsBetweenSummary = batchStepsBetweenSummary

        summaryDir = '{}{}/{}/'.format(summaryDir, self.dateString, self.saveName)
        self.writer = tf.summary.FileWriter(summaryDir, graph=tf.get_default_graph())

    def GetBootstrapTestPerformance(self, sess, testLossOp, bootstrapLossOp):
        numReps = 1000
        confidenceInterval = 0.95
        alpha = (1.0 - confidenceInterval) * 0.5
        lowerBoundIndex = numReps * alpha
        upperBoundIndex = numReps * (1.0 - alpha)

        bootstrapPerformances = np.zeros(N)
        for i in range(N):
            bootstrapPerformances[i] = sess.run(bootstrapLossOp)
        bootstrapPerformances = np.sort(bootstrapPerformances)

        bootstrapPlaceholder = tf.placeholder(dtype=tf.float32, shape=(numreps,), name='bootstrapPlaceholder')
        bootstrapHist =  tf.summary.histogram('Bootstrap Test Performance', bootstrapPlaceholder)
        testScalar    =  tf.summary.scalar('TestLoss', testLossOp)
        histSummary, scalarSummary, pointPerformance = \
            sess.run([bootstrapHist, testScalar, testLossOp],
                     feed_dict={bootstrapPlaceholder: bootstrapPerformances})

        self.writer.add_summary(histSummary, 1)
        self.writer.add_summary(scalarSummary, 1)

        return pointPerformance, bootstrapPerformances[lowerBoundIndex], bootstrapPerformances[upperBoundIndex]

    def SaveModel(self, sess, step, saver):
        """
        Saves the model to path every stepSize steps
        """
        saver.save(sess, self.checkpointDir)
        print('STEP {}: saved model to path {}'.format(step, self.checkpointDir))

    def TrainModel(self, sess, trainUpdateOp, trainLossOp, valdLossOp, testLossOp, bootstrapLossOp):
        tf.summary.scalar('trainingLoss', trainLossOp)
        tf.summary.scalar('validationLoss', valdLossOp)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        self.validationDataSet.InitializeConstantData()
        self.testDataSet.InitializeConstantData()

        extraUpdateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        mergedSummaryOp = tf.summary.merge_all()

        saver = saveModel.restore(sess, savePath)

        bestValidationLoss = math.inf
        bestLossStepIndex = 0

        for batchIndex in range(self.numberOfSteps):
            sess.run([trainUpdateOp, extraUpdateOps], feed_dict=feed_dict)

            if batchIndex % self.batchStepsBetweenSummary == 0:
                summary, trainingLoss, validationLoss = \
                    sess.run([mergedSummaryOp, trainLossOp, valdLossOp], feed_dict=feed_dict)
                print('STEP {}: Training Loss = {}, Validation Loss = {}'.format(batchIndex, trainingLoss, validationLoss))
                self.writer.add_summary(summary, batchIndex)

                if validationLoss < bestValidationLoss:
                    bestLossStepIndex = batchIndex
                    bestValidationLoss = validationLoss
                    self.SaveModel(sess, batchIndex, saver)

        pointTestPerformance, lowerBound, upperBound = \
            self.GetBootstrapTestPerformance(sess=sess,
                                             testLossOp=testLossOp,
                                             bootstrapLossOp=bootstrapLossOp)

        coord.request_stop()
        coord.join(threads)
        print("STEP {}: Best Validation Loss = {}".format(bestLossStepIndex, bestValidationLoss))
        print("Model {} had test performance: {} ({}, {})".format(
            self.saveName,
            pointTestPerformance,
            lowerBound,
            upperBound))