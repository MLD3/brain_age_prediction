import os
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

from utils.config import get

class DataSetNPY(object):
    def __init__(self, numpyDirectory, numpyFileList, labels, reshapeBatches=True):
        """
        Builds a dataset from .npy files in the specified directory.
        Useful if data is too large to fit in memory.
        """
        self.numpyDirectory = numpyDirectory
        self.numpyFileList = numpyFileList
        self.epochsCompleted = 0
        self.labels = labels
        assert numpyFileList.shape[0] == labels.shape[0], (
            'numpyFileList.shape: {}, labels.shape: {}'.format(numpyFileList.shape, labels.shape))
        self.numExamples = numpyFileList.shape[0]
        self.currentStartIndex = 0
        self.reshapeBatches = reshapeBatches
        self.isPreloaded = False
        self.data = None

    def PreloadData(self):
        batchArray = [0] * self.numExamples
        index = 0
        print('Preloading Data...')
        for fileName in self.numpyFileList:
            print('Loading file {} out of {}'.format(index + 1, self.numExamples), end='\r')
            npArray = np.load(self.numpyDirectory + fileName)
            batchArray[index] = npArray
            index += 1

        batchArray = np.array(batchArray)
        if self.reshapeBatches:
            batchArrays = np.expand_dims(batchArrays, len(batchArrays.shape))
        assert batchArray.shape[0] == self.numExamples, 'Batch Shape: {}, num examples: {}'.format(batchArray.shape, self.numExamples)

        self.data = batchArray
        self.isPreloaded = True
        print('Preloading Data Complete.')

    def GetPreloadedData(self):
        return self.data, np.array(labels).reshape((self.numExamples, 1))

    def GetNumpyBatch(self, fileList):
        batchArray = [0] * fileList.shape[0]
        index = 0

        for fileName in fileList:
            npArray = np.load(self.numpyDirectory + fileName)
            batchArray[index] = npArray
            index += 1

        batchArray = np.array(batchArray)
        assert batchArray.shape[0] == fileList.shape[0], 'Batch Shape: {}, File List Shape: {}'.format(batchArray.shape, fileList.shape)
        return batchArray

    def ShuffleData(self):
        permutation = np.arange(self.numExamples)
        np.random.shuffle(permutation)
        self.numpyFileList = self.numpyFileList[permutation]
        self.labels = self.labels[permutation]

    def NextBatch(self, batchSize, shuffle=True):
        assert not self.isPreloaded, 'Cannot shuffle batches if data is preloaded'

        startIndex = self.currentStartIndex

        # Randomly shuffle data for the first epoch
        if self.epochsCompleted == 0 and startIndex == 0 and shuffle:
            self.ShuffleData()

        #This call will finish the current batch and go to the next batch
        if startIndex + batchSize > self.numExamples:
            self.epochsCompleted += 1
            examplesBeforeCutoff = self.numExamples - startIndex
            leftOverFiles = self.numpyFileList[startIndex:]
            leftOverLabels = self.labels[startIndex:]

            if shuffle:
                self.ShuffleData()

            startIndex = 0
            self.currentStartIndex = batchSize - examplesBeforeCutoff
            endIndex = self.currentStartIndex
            newFiles = self.numpyFileList[startIndex:endIndex]
            newLabels = self.labels[startIndex:endIndex]

            fileList = np.concatenate((leftOverFiles, newFiles), axis=0)
            batchArrays = self.GetNumpyBatch(fileList)
            if self.reshapeBatches:
                batchArrays = np.expand_dims(batchArrays, len(batchArrays.shape))
            batchLabels = np.concatenate((leftOverLabels, newLabels), axis=0)
            batchLabels = batchLabels.reshape((batchSize, 1))
            return batchArrays, batchLabels
        else:
            self.currentStartIndex += batchSize
            endIndex = self.currentStartIndex
            fileList = self.numpyFileList[startIndex:endIndex]
            batchArrays = self.GetNumpyBatch(fileList)
            if self.reshapeBatches:
                batchArrays = np.expand_dims(batchArrays, len(batchArrays.shape))
            batchLabels = self.labels[startIndex:endIndex]
            batchLabels = batchLabels.reshape((batchSize, 1))
            return batchArrays, batchLabels

if __name__ == '__main__':
    numpyDirectory = '/Users/psturm/Desktop/Research/brain_age_prediction/Data/'
    numpyFileList = ['a', 'b', 'c', 'd', 'e']
    numpyFileList = np.array([fileName + '.npy' for fileName in numpyFileList])
    labels = np.array([1,2,3,4,5])
    dataset = DataSetNPY(numpyDirectory, numpyFileList, labels)

    for i in range(4):
        data, labels = dataset.NextBatch(3, shuffle=False)
        print('--------Iteration {}--------'.format(i))
        print(data)
        print(labels)
