import nibabel as nib
import numpy as np
import pandas as pd

def ConvertNIItoCSV(inFile, outFile):
    SubjectDataFrame = pd.read_csv('PNC_724_phenotypics.csv')
    for _, row in SubjectDataFrame.iterrows():
        subject = row['Subject']
        age = row['AgeYears']
        print('Saving Subject {}'.format(subject))
        fileName = inFile + subject + '.nii'
        NIIimage = nib.load(fileName)
        imageArray = NIIimage.get_data()
        outFileName = outFile + subject
        np.save(outFileName, imageArray)

if __name__ == '__main__':
    inFileStructural = '/data/psturm/structural/niftiImages'
    inFileFunctional = '/data/psturm/functional/niftiImages/s6_'

    outFileStructural = '/data/psturm/structral/numpyArrays/'
    outFileFunctional = '/data/psturm/functional/numpyArrays/'

    ConvertNIItoCSV(inFileStructural, outFileStructural)
    ConvertNIItoCSV(inFileFunctional, outFileFunctional)
