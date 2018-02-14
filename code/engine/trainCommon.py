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
                        summaryDir,
                        checkpointDir,
                        imagesPL,
                        trainingPL,
                        labelsPL,
                        trainSet,
                        valdSet,
                        testSet,
                        numberOfSteps=get('TRAIN.DEFAULTS.TEST_NB_STEPS'),
                        batchStepsBetweenSummary=500
                        ):

        if not os.path.exists(checkpointDir):
            os.makedirs(checkpointDir)
        self.checkpointDir            = checkpointDir
        self.summaryDir               = summaryDir
        self.numberOfSteps            = numberOfSteps
        self.batchStepsBetweenSummary = batchStepsBetweenSummary

        self.trainLossPlaceholder       = tf.placeholder(tf.float32, shape=(), name='trainLossPlaceholder')
        self.validationLossPlaceholder  = tf.placeholder(tf.float32, shape=(), name='validationLossPlaceholder')
        self.trainSummary               = tf.summary.scalar('trainingLoss', self.trainLossPlaceholder)
        self.validationSummary          = tf.summary.scalar('validationLoss', self.validationLossPlaceholder)
        self.imagesPL      = imagesPL
        self.trainingPL    = trainingPL
        self.labelsPL      = labelsPL
        self.trainSet      = trainSet
        self.valdSet       = valdSet
        self.testSet       = testSet

    def GetFeedDict(self, sess, setType='train'):
        if setType == 'train':
            images, labels = self.trainSet.NextBatch(sess)
            training = True
        elif setType == 'vald':
            images, labels = self.valdSet.NextBatch(sess)
            training = False
        elif setType == 'test':
            images, labels = self.testSet.NextBatch(sess)
            training = False
        return {
            self.imagesPL: images,
            self.labelsPL: labels,
            self.trainingPL: training
        }

    def GetPerformanceThroughSet(self, sess, lossOp, setType='vald', numberIters=75):
        accumulatedLoss = 0
        for i in range(numberIters):
            currentLoss = sess.run(lossOp, feed_dict=self.GetFeedDict(sess, setType=setType))
            accumulatedLoss += currentLoss
        accumulatedLoss = accumulatedLoss / numberIters
        return accumulatedLoss

    def SaveModel(self, sess, step, saver, path):
        """
        Saves the model to path every stepSize steps
        """
        saver.save(sess, path)
        print('STEP {}: saved model to path {}'.format(step, path), end='\r')

    def TrainModel(self, sess, updateOp, lossOp, name, restore=False):
        writer = tf.summary.FileWriter('{}{}/'.format(self.summaryDir, name))

        # Initialize relevant variables
        sess.run(tf.global_variables_initializer())

        # Collect summary and graph update operations
        extraUpdateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if restore:
            savePath = '{}{}/'.format(self.checkpointDir, name)
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            # Restore a model if it exists in the indicated directory
            saver = saveModel.restore(sess, savePath)
        # else:
            # print('Reading saved models is disabled. Training from scratch...')

        bestValidationLoss = math.inf
        bestLossStepIndex = 0

        for batchIndex in range(self.numberOfSteps):
            trainingLoss, _, _ = sess.run([lossOp, updateOp, extraUpdateOps],
                                            feed_dict=self.GetFeedDict(sess))

            if batchIndex % self.batchStepsBetweenSummary == 0:
                validationLoss = self.GetPerformanceThroughSet(sess, lossOp)

                # print('STEP {}: Training Loss = {}, Validation Loss = {}'.format(
                            # batchIndex,
                            # trainingLoss,
                            # validationLoss),
                            # end='\r')
                writer.add_summary(
                    sess.run(
                        self.trainSummary,
                        feed_dict={
                            self.trainLossPlaceholder: trainingLoss
                        }),
                    batchIndex)
                writer.add_summary(
                    sess.run(
                        self.validationSummary,
                        feed_dict={
                            self.validationLossPlaceholder: validationLoss
                        }),
                    batchIndex)

                if validationLoss < bestValidationLoss:
                    bestLossStepIndex = batchIndex
                    bestValidationLoss = validationLoss
                    if restore:
                        self.SaveModel(sess, batchIndex, saver, savePath)

        testLoss = self.GetPerformanceThroughSet(sess, lossOp, setType='test')
        writer.close()

        # print("STEP {}: Best Validation Loss = {}".format(bestLossStepIndex, bestValidationLoss))
        # print("Model had test performance: {}".format(testLoss))
        return bestValidationLoss, testLoss

    def CompareRuns(self, sess, updateOps, lossOp, names):
        graphWriter = tf.summary.FileWriter(self.summaryDir, graph=tf.get_default_graph())
        graphWriter.close()

        # Start the threads to read in data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        bestValidationLoss = math.inf
        bestTestLoss = math.inf
        bestIndex = 0
        for i in range(len(trainUpdateOps)):
            print('============TRAINING MODEL {}============'.format(names[i]))
            validationLoss, testLoss = self.TrainModel(sess,
                                                       updateOps[i],
                                                       lossOp,
                                                       names[i],
                                                       restore=True)
            if validationLoss < bestValidationLoss:
                bestValidationLoss = validationLoss
                bestTestLoss = testLoss
                bestIndex = i
        print('Best model was: {}'.format(names[bestIndex]))
        print('Validation loss: {}'.format(bestValidationLoss))
        print('Test loss: {}'.format(bestTestLoss))
        coord.request_stop()
        coord.join(threads)

    def RepeatTrials(self, sess, updateOp, lossOp, name, numIters=10):
        print('TRAINING MODEL {}'.format(name))
        graphWriter = tf.summary.FileWriter(self.summaryDir, graph=tf.get_default_graph())
        graphWriter.close()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        valdLosses = []
        testLosses = []
        for i in range(numIters):
            # print('=========Training iteration {}========='.format(i))
            validationLoss, testLoss = self.TrainModel(sess,
                                                       updateOp,
                                                       lossOp,
                                                       '{}/run_{}'.format(name, i))
            valdLosses.append(validationLoss)
            testLosses.append(testLoss)
        print('Average Validation Performance: {} +- {}'.format(np.mean(valdLosses), np.std(valdLosses)))
        print('Average Test Performance: {} +- {}'.format(np.mean(testLosses), np.std(testLosses)))
        coord.request_stop()
        coord.join(threads)
