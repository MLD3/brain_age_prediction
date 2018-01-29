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

def NPYToBinaryDataset(inFile, outFile, SubjectDataFrame):
    numRows = SubjectDataFrame.shape[0]
    brainImages = np.zeros((numRows, 145, 145, 145))
    ages = np.zeros((numRows))
    index = 0
    for _, row in SubjectDataFrame.iterrows():
        index += 1
        subject = row['Subject']
        age = row['AgeYears']
        print('Reading subject {}, {} out of {}'.format(subject, index, numRows), end='\r')
        fileName = '{}{}.npy'.format(inFile, subject)
        image = np.load(fileName)
        image = np.pad(image, [(0,0), (24,0), (0,0), (24,0)], mode='constant')

        brainImages[index - 1, :, :, :] = image
        ages[index - 1] = age

    indices = np.random.permutation(numRows)
    testIndices = indices[:75]
    valdIndices = indices[75:150]
    trainIndices = indices[150:]

    flattenedImages = brainImages.reshape((numRows, 145*145*145))
    flattenedImages = np.insert(flattenedImages, 0, ages, axis=1)
    print(flattenedImages.shape)
    print('Saving test set...')
    flattenedImages[testIndices].tofile('{}structural_test.bin'.format(outFile))
    print('Saving validation set...')
    flattenedImages[valdIndices].tofile('{}structural_vald.bin'.format(outFile))
    print('Saving training set...')
    flattenedImages[trainIndices].tofile('{}structural_train.bin'.format(outFile))
    flattenedImages = None

    trainingImages = brainImages[trainIndices]
    trainingAges = ages[trainIndices]
    numTraining = numRows - 150
    desiredDim = 145

    flattenedImages = np.zeros((numTraining * 121, 145 * 145 + 1))
    index = 0
    for j in range(numTraining):
        print('Slicing image {} of {}, x axis'.format(j, numTraining), end='\r')
        currentImage = trainingImages[j, :, :, :]
        currentAge = trainingAges[j]
        for i in range(121):
            xSlice = currentImage[i, :, :]
            xSlice = np.pad(xSlice, [(heightPadding, 0), (depthPadding, 0)], mode='constant')
            assert xSlice.shape == (desiredDim, desiredDim), 'Shape {} is not correct'.format(xSlice.Shape)
            xSlice = xSlice.flatten()
            xSlice = np.insert(xSlice, 0, currentAge)
            flattenedImages[index, :] = xSlice
            index += 1
    print('Saving x axis training set...')
    flattenedImages.tofile('{}xAxisSlices/train.bin'.format(outFile))

    flattenedImages = np.zeros((numTraining * 145, 145 * 145 + 1))
    index = 0
    for j in range(numTraining):
        print('Slicing image {} of {}, y axis'.format(j, numTraining), end='\r')
        currentImage = trainingImages[j, :, :, :]
        currentAge = trainingAges[j]
        for i in range(145):
            ySlice = currentImage[:, i, :]
            ySlice = np.pad(ySlice, [(widthPadding, 0), (depthPadding, 0)], mode='constant')
            assert ySlice.shape == (desiredDim, desiredDim), 'Shape {} is not correct'.format(ySlice.Shape)
            ySlice = ySlice.flatten()
            ySlice = np.insert(ySlice, 0, currentAge)
            flattenedImages[index, :] = ySlice
            index += 1
    print('Saving y axis training set...')
    flattenedImages.tofile('{}yAxisSlices/train.bin'.format(outFile))

    flattenedImages = np.zeros((numTraining * 121, 145 * 145 + 1))
    index = 0
    for j in range(numTraining):
        print('Slicing image {} of {}, z axis'.format(j, numTraining), end='\r')
        currentImage = trainingImages[j, :, :, :]
        currentAge = trainingAges[j]
        for i in range(121):
            zSlice = currentImage[:, :, i]
            zSlice = np.pad(zSlice, [(widthPadding, 0), (heightPadding, 0)], mode='constant')
            assert zSlice.shape == (desiredDim, desiredDim), 'Shape {} is not correct'.format(zSlice.Shape)
            zSlice = zSlice.flatten()
            zSlice = np.insert(zSlice, 0, currentAge)
            flattenedImages[index, :] = zSlice
            index += 1
    print('Saving z axis training set...')
    flattenedImages.tofile('{}zAxisSlices/train.bin'.format(outFile))


if __name__ == '__main__':
    SubjectDataFrame = pd.read_csv('/data/psturm/PNC_724_phenotypics.csv')
    NPYToBinaryDataset(inFile='/data/psturm/structural/numpyArrays/', outFile='/data/psturm/structural/', SubjectDataFrame=SubjectDataFrame)
