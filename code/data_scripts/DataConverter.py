import nibabel as nib
import numpy as np
import pandas as pd

def ConvertNIItoCSV(inFile, outFile, SubjectDataFrame):
    for _, row in SubjectDataFrame.iterrows():
        subject = row['Subject']
        print('Saving Subject {}'.format(subject))
        fileName = inFile + str(subject) + '.nii'
        NIIimage = nib.load(fileName)
        imageArray = NIIimage.get_data()
        outFileName = outFile + str(subject)
        np.save(outFileName, imageArray)

def convertCSVToNPY(inFile, outFile, SubjectDataFrame):
    for _, row in SubjectDataFrame.iterrows():
        subject = row['Subject']
        print('Saving Subject {}'.format(subject))
        fileName = inFile + str(subject) + '.csv'
        df = pd.read_csv(fileName, header=None)
        npArray = df.as_matrix()
        outFileName = outFile + str(subject)
        np.save(outFileName, npArray)

def ConvertNPYToBinary(inFile, outFile, SubjectDataFrame, maxDims=121, midChar='x'):
    numRows = SubjectDataFrame.shape[0]
    numExamples = numRows * maxDims
    exampleWidth = 145*145 + 1
    accumulatedArrays = np.zeros((numExamples, exampleWidth))

    index = 0
    j = 0
    print('Converting data in file {}'.format(inFile))
    for _, row in SubjectDataFrame.iterrows():
        index += 1
        subject = row['Subject']
        age = row['AgeYears']
        print('Reading subject {}, {} out of {}'.format(subject, index, numRows), end='\r')
        for i in range(maxDims):
            fileName = "{}{}_{}_{}.npy".format(inFile, subject, midChar, i)
            npArray = np.load(fileName)
            npArray = npArray.flatten()
            npArray = npArray.astype(np.float64)
            npArray = np.insert(npArray, 0, age)
            accumulatedArrays[j, :] = npArray
            j += 1

    print("Shape of accumulated arrays: {}".format(accumulatedArrays.shape))
    indices = np.random.permutation(accumulatedArrays.shape[0])
    testSet = accumulatedArrays[indices[:5000]]
    print("Test set shape: {}".format(testSet.shape))
    validationSet = accumulatedArrays[indices[5000:10000]]
    print("Validation set shape: {}".format(validationSet.shape))
    trainingSet = accumulatedArrays[indices[10000:]]
    print("Training set shape: {}".format(trainingSet.shape))

    print('Writing test set to file...')
    testSet.tofile('{}_test.bin'.format(outFile))
    print('Writing validation set to file...')
    validationSet.tofile('{}_vald.bin'.format(outFile))
    print('Writing training set to file...')
    trainingSet.tofile('{}_train.bin'.format(outFile))

def SpliceNIIFilesAlongAxes(inFile, outFile, SubjectDataFrame):
    for _, row in SubjectDataFrame.iterrows():
        subject = row['Subject']
        print('Splicing Subject {}'.format(subject))
        fileName = inFile + str(subject) + '.nii'
        NIIimage = nib.load(fileName)
        imageArray = NIIimage.get_data()

        desiredDim = 145

        currentWidth, currentHeight, currentDepth = imageArray.shape
        widthPadding = desiredDim - currentWidth
        heightPadding = desiredDim - currentHeight
        depthPadding = desiredDim - currentDepth

        xSlicesName = '{}xAxisSlices/'.format(outFile)
        for i in range(currentWidth):
            xSlice = imageArray[i, :, :]
            xSlice = np.pad(xSlice, [(heightPadding, 0), (depthPadding, 0)], mode='constant')
            assert xSlice.shape == (desiredDim, desiredDim), 'Shape {} is not correct'.format(xSlice.Shape)
            np.save('{}{}_x_{}'.format(xSlicesName, subject, i), xSlice)

        ySlicesName = '{}yAxisSlices/'.format(outFile)
        for i in range(currentHeight):
            ySlice = imageArray[:, i, :]
            ySlice = np.pad(ySlice, [(widthPadding, 0), (depthPadding, 0)], mode='constant')
            assert ySlice.shape == (desiredDim, desiredDim), 'Shape {} is not correct'.format(ySlice.Shape)
            np.save('{}{}_y_{}'.format(ySlicesName, subject, i), ySlice)

        zSlicesName = '{}zAxisSlices/'.format(outFile)
        for i in range(currentDepth):
            zSlice = imageArray[:, :, i]
            zSlice = np.pad(zSlice, [(widthPadding, 0), (heightPadding, 0)], mode='constant')
            assert zSlice.shape == (desiredDim, desiredDim), 'Shape {} is not correct'.format(zSlice.Shape)
            np.save('{}{}_z_{}'.format(zSlicesName, subject, i), zSlice)

if __name__ == '__main__':
    SubjectDataFrame = pd.read_csv('/data/psturm/PNC_724_phenotypics.csv')
    ConvertNPYToBinary(inFile='/data/psturm/structural/yAxisSlices/', outFile='/data/psturm/structural/yAxisSlices/dataSet', SubjectDataFrame=SubjectDataFrame, maxDims=145, midChar='y')
    ConvertNPYToBinary(inFile='/data/psturm/structural/zAxisSlices/', outFile='/data/psturm/structural/zAxisSlices/dataSet', SubjectDataFrame=SubjectDataFrame, maxDims=121, midChar='z')
