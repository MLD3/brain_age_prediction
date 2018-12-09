import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
import re
from sklearn.model_selection import train_test_split, KFold
from utils import saveModel
from utils.config import get
from placeholders.shared_placeholders import *
from datetime import datetime
from utils.args import *
from tensorflow.contrib.memory_stats import BytesInUse


class PrintOps(object):
    """
    This class wraps an arbitrary set of operations and supports
    printing those operations to tensorboard summary files.
    """
    def flatten(self, tensor):
        name = tensor.name.split('/')
        if len(name) > 5:
            name = '{}_{}_{}'.format(name[3], name[4], name[5])
        else:
            name = re.sub('[^0-9a-zA-Z]+', '_', tensor.name)
        shape = tensor.get_shape().as_list()
        dim = np.prod(shape)
        flattened = tf.reshape(tensor, (dim,), name=name)
        return flattened

    def __init__(self, ops, updateOps, names, gradients):
        self.ops = ops
        self.updateOps = updateOps
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

        with tf.variable_scope('IndividualGradients'):
            self.gradients = [self.flatten(grad) for grad in gradients]
            concatGradients = tf.concat(self.gradients, axis=0, name='ConcatenatedGradients')
            gradientHistograms = [tf.summary.histogram(grad.name + '_hist', grad) for grad in self.gradients]
            gradientMeans = [tf.summary.scalar(grad.name + '_mean', tf.reduce_mean(grad)) for grad in self.gradients]
        with tf.variable_scope('ConcatenatedGradient'):
            concatHist = tf.summary.histogram('ConcatenatedGradients_hist', concatGradients)
            concatMean = tf.summary.scalar('ConcatenatedGradients_mean', tf.reduce_mean(concatGradients))
        self.gradientSummary = tf.summary.merge(gradientHistograms + gradientMeans + [concatHist] + [concatMean])

class ModelTrainer(object):
    """
    This class supports training an arbitrary model with a gradient-based
    minimization algorithm.
    """
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
                        numberOfSteps=501,
                        batchStepsBetweenSummary=2500,
                        phenotypicsPL=None
                        ):
        if not os.path.exists(checkpointDir) and GlobalOpts.validationDir is None:
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
        self.phenotypicsPL = phenotypicsPL

    def GetFeedDict(self, sess, setType='train', setIndex=0):
        if self.phenotypicsPL is not None:
            if setType == 'train':
                images, labels, phenotypes = self.trainSet[setIndex].NextBatch(sess)
                training = True
            elif setType == 'vald':
                images, labels, phenotypes = self.valdSet[setIndex].NextBatch(sess)
                training = False
            elif setType == 'test':
                images, labels, phenotypes = self.testSet[setIndex].NextBatch(sess)
                training = False
            return {
                self.imagesPL: images,
                self.labelsPL: labels,
                self.trainingPL: training,
                self.phenotypicsPL: phenotypes
            }
        else:
            if setType == 'train':
                images, labels = self.trainSet[setIndex].NextBatch(sess)
                training = True
            elif setType == 'vald':
                images, labels = self.valdSet[setIndex].NextBatch(sess)
                training = False
            elif setType == 'test':
                images, labels = self.testSet[setIndex].NextBatch(sess)
                training = False
            return {
                self.imagesPL: images,
                self.labelsPL: labels,
                self.trainingPL: training
            }

    def GetPerformanceThroughSet(self, sess, printOps, setType='vald', batchTrainFeedDict=None, setIndex=0):
        sess.run(tf.local_variables_initializer())
        accumulatedOps = sess.run(printOps.ops)
        if setType == 'vald':
            numberIters = (self.valdSet[setIndex].maxItemsInQueue + self.valdSet[setIndex].batchSize - 1)\
                           //self.valdSet[setIndex].batchSize
            print(numberIters)
            assert(numberIters > 0)
        elif setType == 'test':
            numberIters = self.testSet[setIndex].maxItemsInQueue
        elif setType == 'train':
            numberIters = 1

        for i in range(numberIters):
            if setType == 'vald':
                print(self.valdSet[setIndex])
                feed_dict = self.GetFeedDict(sess, setType=setType, setIndex=setIndex)
            elif setType == 'test':
                feed_dict = self.GetFeedDict(sess, setType=setType, setIndex=setIndex)
            elif setType == 'train':
                feed_dict = batchTrainFeedDict
            sess.run(printOps.updateOps, feed_dict=feed_dict)
            print(i)

        if setType == 'vald':
            self.valdSet[setIndex].RefreshNumEpochs()

        accumulatedOps = sess.run(printOps.ops)
        summaryFeedDict = {}
        opValueDict = {}
        for i in range(len(printOps.ops)):
            opValueDict[printOps.names[i]] = accumulatedOps[i]
            if setType == 'vald':
                summaryFeedDict[printOps.valdPlaceholders[i]] = accumulatedOps[i]
            elif setType == 'train':
                summaryFeedDict[printOps.trainPlaceholders[i]] = accumulatedOps[i]

        return opValueDict, summaryFeedDict

    def SaveModel(self, sess, step, saver, path):
        """
        Saves the model to path every stepSize steps
        """
        saver.save(sess, path)
        print('STEP {}: saved model to path {}'.format(step, path), end='\r')

    def TrainModel(self, sess, updateOp, printOps, name, setIndex=0):
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
        stepsSinceLastBest = 0
        maxStepsBeforeStop = 50000

        for batchIndex in range(self.numberOfSteps):
            batchTrainFeedDict = self.GetFeedDict(sess, setIndex=setIndex)

            if batchIndex % self.batchStepsBetweenSummary != 0:
                _, _ = sess.run([updateOp, extraUpdateOps], feed_dict=batchTrainFeedDict)
            else:
                _, _, gradSummary = sess.run([updateOp, extraUpdateOps, printOps.gradientSummary], feed_dict=batchTrainFeedDict)
                writer.add_summary(gradSummary, batchIndex)

                opValueDict, summaryFeedDict = self.GetPerformanceThroughSet(sess, printOps,
                                    setType='train', batchTrainFeedDict=batchTrainFeedDict, setIndex=setIndex)
                writer.add_summary(
                    sess.run(
                        printOps.mergedTrainSummary,
                        feed_dict=summaryFeedDict),
                    batchIndex)
                print("==============Train Set Operations, Step {}==============".format(batchIndex))
                for opName in opValueDict:
                    print('{}: {}'.format(opName, opValueDict[opName]))

                opValueDict, summaryFeedDict = self.GetPerformanceThroughSet(sess, printOps, setIndex=setIndex)
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
                    stepsSinceLastBest = 0
                else:
                    stepsSinceLastBest += self.batchStepsBetweenSummary
                    if stepsSinceLastBest >= maxStepsBeforeStop:
                        break

        saveModel.restore(sess, saver, savePath)
        testOpValueDict, _ = self.GetPerformanceThroughSet(sess, printOps, setType='test', setIndex=setIndex)
        writer.close()

        return bestValdOpDict, testOpValueDict

    def RepeatTrials(self, sess, updateOp, printOps, name, numIters=5):
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
            # self.valdSet[i % 5].PreloadData()
            print('=========Training iteration {}========='.format(i))
            valdOpDict, testOpDict = self.TrainModel(sess,
                                                       updateOp,
                                                       printOps,
                                                       '{}/run_{}'.format(name, i),
                                                       i % 5)
            for opName in printOps.names:
                bestValdOpDict[opName].append(valdOpDict[opName])
                bestTestOpDict[opName].append(testOpDict[opName])
            # self.valdSet[i % 5].UnloadData()

        outputFile = open('{}performance.txt'.format(self.summaryDir), 'w')

        print("==============Validation Set Operations, Best==============")
        outputFile.write("==============Validation Set Operations, Best==============\n")
        for opName in bestValdOpDict:
            outputString = '{}: {} +- {}\t{}'.format(opName, np.mean(bestValdOpDict[opName]), np.std(bestValdOpDict[opName]), bestValdOpDict[opName])
            print(outputString)
            outputFile.write(outputString + '\n')
        print("==============Test Set Operations, Best==============")
        outputFile.write("==============Test Set Operations, Best==============\n")
        for opName in bestTestOpDict:
            outputString = '{}: {} +- {}\t{}'.format(opName, np.mean(bestTestOpDict[opName]), np.std(bestTestOpDict[opName]), bestTestOpDict[opName])
            print(outputString)
            outputFile.write(outputString + '\n')
        outputFile.close()
        coord.request_stop()
        coord.join(threads)

    def ValidateModel(self, sess, updateOp, printOps, name, numIters=5):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('Model: {}'.format(name))
        bestValdOpDict = {}
        bestTestOpDict = {}
        for opName in printOps.names:
            bestValdOpDict[opName] = []
            bestTestOpDict[opName] = []

        for i in range(numIters):
            sess.run(tf.global_variables_initializer())
            extraUpdateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            saver = tf.train.Saver()
            savePath = '{}run_{}/'.format(GlobalOpts.validationDir, i)
            saveModel.restore(sess, saver, savePath)

            valdOpDict, _ = self.GetPerformanceThroughSet(sess, printOps, setType='vald')
            testOpDict, _ = self.GetPerformanceThroughSet(sess, printOps, setType='test')
            for opName in printOps.names:
                bestValdOpDict[opName].append(valdOpDict[opName])
                bestTestOpDict[opName].append(testOpDict[opName])

        print("==============Validation Set Operations, Best==============")
        for opName in bestValdOpDict:
            outputString = '{}: {} +- {}\t{}'.format(opName, np.mean(bestValdOpDict[opName]), np.std(bestValdOpDict[opName]), bestValdOpDict[opName])
            print(outputString)
        print("==============Test Set Operations, Best==============")
        for opName in bestTestOpDict:
            outputString = '{}: {} +- {}\t{}'.format(opName, np.mean(bestTestOpDict[opName]), np.std(bestTestOpDict[opName]), bestTestOpDict[opName])
            print(outputString)
        coord.request_stop()
        coord.join(threads)

    def getPatientPerformances(self, sess, predictionOp, name, numIters=1):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        nameOp, labelOp, imageOp = [], [], []
        for i in range(5):
            nameOp.append(self.testSet[i].dequeueOp)
            labelOp.append(self.testSet[i].labelBatchOperation)
            imageOp.append(self.testSet[i].imageBatchOperation)
        numberIters = self.testSet[0].maxItemsInQueue
        predictedAges = np.zeros((numIters, numberIters))
        absoluteErrors = np.zeros((numIters, numberIters))
        squaredErrors = np.zeros((numIters, numberIters))
        trueAges = np.zeros((numberIters, ))
        name_arr = []
        print('Model: {}'.format(name))

        print('SUBJECT ID\tTRUE AGE\tPREDICTED AGE')
        for i in range(numIters):
            sess.run(tf.global_variables_initializer())
            extraUpdateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            saver = tf.train.Saver()
            savePath = '{}run_{}/'.format(GlobalOpts.validationDir, i)
            saveModel.restore(sess, saver, savePath)

            for j in range(numberIters):
                images, labels, names = sess.run([imageOp[i], labelOp[i], nameOp[i]])
                feed_dict = {
                    self.imagesPL: images,
                    self.labelsPL: labels,
                    self.trainingPL: False
                }
                names = names[0].decode('UTF-8')
                if i == 0:
                    name_arr.append(names)
                labels = labels[0, 0]
                predictions = sess.run([predictionOp], feed_dict=feed_dict)
                predictions = np.squeeze(predictions[0])
                trueAges[j] = labels
                predictedAges[i, j] = predictions
                absoluteErrors[i, j] = np.abs(predictions - labels)
                squaredErrors[i, j] = np.square(predictions - labels)
#                print('{}\t{:.4f}\t{:.4f}'.format(names, labels, predictions))
        df = pd.DataFrame(data = {
            'Subject': np.array(name_arr),
            'TrueAge': trueAges,
            'Run_1': predictedAges[0, :],
            'Run_2': predictedAges[1, :],
            'Run_3': predictedAges[2, :],
            'Run_4': predictedAges[3, :],
            'Run_5': predictedAges[4, :],
            'MinPredicted': np.min(predictedAges, axis=0),
            'MaxPredicted': np.max(predictedAges, axis=0),
            'sdPredicted': np.std(predictedAges, axis=0),
            'AE_1': predictedAges[0, :],
            'AE_2': predictedAges[1, :],
            'AE_3': predictedAges[2, :],
            'AE_4': predictedAges[3, :],
            'AE_5': predictedAges[4, :],
            'SE_1': predictedAges[0, :],
            'SE_2': predictedAges[1, :],
            'SE_3': predictedAges[2, :],
            'SE_4': predictedAges[3, :],
            'SE_5': predictedAges[4, :],

        })
        # print(df)
        MAE = np.mean(absoluteErrors, axis=1)
        MSE = np.mean(squaredErrors, axis=1)
        print("MAE: " + str(MAE))
        print("MSE: " + str(MSE))
        # for j in range(numberIters):
        #     print('{}\t{}\t{}'.format(name_arr[j], trueAges[j], predictedAges[:, j]))
        df.to_csv('{}{}.csv'.format("/home/hyhuang/brain_age_prediction/reports/",name), index=False)
        coord.request_stop()
        coord.join(threads)