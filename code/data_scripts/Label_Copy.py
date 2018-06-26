import pandas as pd
import numpy as np
from skimage.measure import block_reduce

inputFile = 'PNC/labels/'
outputFile = 'PNC_AUGMENTED/general_labels/'

df = pd.read_csv('PNC/PNC_724_phenotypics.csv')

for _, row in df.iterrows():
    subject = row['Subject']
    print('Reading subject {}'.format(subject), end='\r')
    label = np.load('{}{}.npy'.format(inputFile, subject))
    np.save('{}{}{}.npy'.format(outputFile, subject, "1"), label)
    np.save('{}{}{}.npy'.format(outputFile, subject, "2"), label)
    np.save('{}{}{}.npy'.format(outputFile, subject, "3"), label)
    np.save('{}{}{}.npy'.format(outputFile, subject, "4"), label)
    np.save('{}{}{}.npy'.format(outputFile, subject, "5"), label)
    np.save('{}{}.npy'.format(outputFile, subject), label)