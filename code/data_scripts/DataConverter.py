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

if __name__ == '__main__':
    SubjectDataFrame = pd.read_csv('/data/psturm/PNC_724_phenotypics.csv')
    inFileStructural = '/data/psturm/structural/niftiImages/'
    inFileFunctional = '/data/psturm/functional/niftiImages/s6_'

    outFileStructural = '/data/psturm/structural/numpyArrays/'
    outFileFunctional = '/data/psturm/functional/numpyArrays/'

    ConvertNIItoCSV(inFileStructural, outFileStructural, SubjectDataFrame)
    ConvertNIItoCSV(inFileFunctional, outFileFunctional, SubjectDataFrame)
