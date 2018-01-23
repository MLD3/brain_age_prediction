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

def ConvertNPYToBinary(inFile, outFile, SubjectDataFrame):
    accumulatedArrays = []
    for _, row in SubjectDataFrame.iterrows():
        subject = row['Subject']
        age = row['AgeYears']
        print('Reading Subject {}'.format(subject))
        fileName = inFile + str(subject) + '.npy'
        npArray = np.load(fileName)
        npArray = npArray.flatten()
        npArray = npArray.astype(np.float32)
        npArray = np.insert(npArray, 0, age)
        accumulatedArrays.append(npArray)
    print('Writing Binary File...')
    accumulatedArrays = np.array(accumulatedArrays)
    accumulatedArrays.tofile('{}.bin'.format(outFile))

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
            np.save('{}{}_x_{}'.format(xSlicesName, subject, i))

        ySlicesName = '{}yAxisSlices/'.format(outFile)
        for i in range(currentHeight):
            ySlice = imageArray[:, i, :]
            ySlice = np.pad(ySlice, [(widthPadding, 0), (depthPadding, 0)], mode='constant')
            assert ySlice.shape == (desiredDim, desiredDim), 'Shape {} is not correct'.format(ySlice.Shape)
            np.save('{}{}_y_{}'.format(ySlicesName, subject, i))

        zSlicesName = '{}zAxisSlices/'.format(outFile)
        for i in range(currentDepth):
            zSlice = imageArray[:, :, i]
            zSlice = np.pad(zSlice, [(widthPadding, 0), (heightPadding, 0)], mode='constant')
            assert zSlice.shape == (desiredDim, desiredDim), 'Shape {} is not correct'.format(zSlice.Shape)
            np.save('{}{}_z_{}'.format(zSlicesName, subject, i))

if __name__ == '__main__':
    SubjectDataFrame = pd.read_csv('/data/psturm/PNC_724_phenotypics.csv')
    inFileStructural = '/data/psturm/structural/niftiImages/'
    outFileStructural = '/data/psturm/structural/'
    SpliceNIIFilesAlongAxes(inFileStructural, outFileStructural, SubjectDataFrame)
