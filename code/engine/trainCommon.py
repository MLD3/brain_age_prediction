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

class PrintOps(object):
    def __init__(self, ops, names):
        self.ops = ops
        self.names = names
        self.valdPlaceholders = [None] * len(names)
        self.valdSummaries = [None] * len(names)
        self.trainPlaceholders = [None] * len(names)
        self.trainSummaries = [None] * len(names)
        with tf.variable_scope('SummaryOps'):
            for i in range(len(self.names)):
                self.valdPlaceholders[i] = tf.placeholder(tf.float32, shape=(), name='{}ValdPlaceholder'.format(self.names[i]))
                self.valdSummaries[i] = tf.summary.scalar('{}Vald'.format(self.names[i]), self.valdPlaceholders[i])
                self.trainPlaceholders[i] = tf.placeholder(tf.float32, shape=(), name='{}TrainPlaceholder'.format(self.names[i]))
                self.trainSummaries[i] = tf.summary.scalar('{}Train'.format(self.names[i]), self.trainPlaceholders[i])
            self.mergedValdSummary = tf.summary.merge(self.valdSummaries)
            self.mergedTrainSummary = tf.summary.merge(self.trainSummaries)

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

    def GetPerformanceThroughSet(self, sess, printOps, setType='vald', numberIters=75): #TODO: CHANGE NUMBER ITERS DYNAMICALLY
        accumulatedOps = np.zeros(shape=(len(printOps.ops),))

        for i in range(numberIters):
            opValues = sess.run(printOps, feed_dict=self.GetFeedDict(sess, setType=setType))
            accumulatedOps += opValues

        accumulatedOps = accumulatedOps / numberIters
        summaryFeedDict = {}
        opValueDict = {}
        for i in range(len(printOps.ops)):
            opValueDict[printOps.names[i]] = accumulatedOps[i]
            summaryFeedDict[printOps.placeholders[i]] = accumulatedOps[i]

        return opValueDict, summaryFeedDict

    def SaveModel(self, sess, step, saver, path):
        """
        Saves the model to path every stepSize steps
        """
        saver.save(sess, path)
        print('STEP {}: saved model to path {}'.format(step, path), end='\r')

    def TrainModel(self, sess, updateOp, printOps, name):
        writer = tf.summary.FileWriter('{}{}/'.format(self.summaryDir, name))

        # Initialize relevant variables
        sess.run(tf.global_variables_initializer())

        # Collect summary and graph update operations
        extraUpdateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        saver = tf.train.Saver()
        savePath = '{}{}/'.format(self.checkpointDir, name)
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        bestValidationLoss = math.inf
        bestValdOpDict = {}
        bestLossStepIndex = 0

        for batchIndex in range(self.numberOfSteps):
            _, _ = sess.run([updateOp, extraUpdateOps], feed_dict=self.GetFeedDict(sess))

            if batchIndex % self.batchStepsBetweenSummary == 0:
                opValueDict, summaryFeedDict = self.GetPerformanceThroughSet(sess, printOps, setType='train')
                writer.add_summary(
                    sess.run(
                        printOps.mergedTrainSummary,
                        feed_dict=summaryFeedDict),
                    batchIndex)
                print("==============Train Set Operations, Step {}==============".format(batchIndex))
                for opName in opValueDict:
                    print('{}: {}'.format(opName, opValueDict[opName]))

                opValueDict, summaryFeedDict = self.GetPerformanceThroughSet(sess, printOps)
                writer.add_summary(
                    sess.run(
                        printOps.mergedValdSummary,
                        feed_dict=summaryFeedDict),
                    batchIndex)
                print("==============Validation Set Operations, Step {}==============".format(batchIndex))
                for opName in opValueDict:
                    print('{}: {}'.format(opName, opValueDict[opName]))

                validationLoss = opValueDict['loss']
                if validationLoss < bestValidationLoss:
                    bestLossStepIndex = batchIndex
                    bestValidationLoss = validationLoss
                    bestValdOpDict = opValueDict
                    self.SaveModel(sess, batchIndex, saver, savePath)

        saveModel.restore(sess, saver, savePath)
        testOpValueDict, _ = self.GetPerformanceThroughSet(sess, printOps, setType='test')
        writer.close()

        return bestValdOpDict, testOpValueDict

    def RepeatTrials(self, sess, updateOp, printOps, name, numIters=10):
        print('TRAINING MODEL {}'.format(name))
        graphWriter = tf.summary.FileWriter(self.summaryDir, graph=tf.get_default_graph())
        graphWriter.close()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        bestValdOpDict = {}
        bestTestOpDict = {}
        for opName in printOps.names:
            bestValdOpDict[opName] = []
            bestTestOpDict[opName] = []

        for i in range(numIters):
            print('=========Training iteration {}========='.format(i))
            valdOpDict, testOpDict = self.TrainModel(sess,
                                                       updateOp,
                                                       printOps,
                                                       '{}/run_{}'.format(name, i))
            for opName in printOps.names:
                bestValdOpDict[opName].append(valdOpDict[opName])
                bestTestOpDict[opName].append(testOpDict[opName])

        outputFile = open('{}performance.txt'.format(self.summaryDir), 'w')

        print("==============Validation Set Operations, Best==============")
        outputFile.write("==============Validation Set Operations, Best==============")
        for opName in bestValdOpDict:
            outputString = '{}: {} +- {}'.format(opName, np.mean(bestValdOpDict[opName]), np.std(bestValdOpDict[opName]))
            print(outputString)
            outputFile.write(outputString)
        print("==============Test Set Operations, Best==============")
        outputFile.write("==============Test Set Operations, Best==============")
        for opName in bestTestOpDict:
            outputString = '{}: {} +- {}'.format(opName, np.mean(bestTestOpDict[opName]), np.std(bestTestOpDict[opName]))
            print(outputString)
            outputFile.write(outputString)
        outputFile.close()
        coord.request_stop()
        coord.join(threads)
