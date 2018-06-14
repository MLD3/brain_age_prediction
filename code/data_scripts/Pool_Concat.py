import pandas as pd
import numpy as np
from skimage.measure import block_reduce

avg_file = 'PNC/structural/avgpool3x3x3/'
max_file = 'PNC/structural/maxpoolArrays/'
outputFile = 'PNC_AUGMENTED/pool_concat/'

df = pd.read_csv('PNC/PNC_724_phenotypics.csv')


for _, row in df.iterrows():
    subject = row['Subject']
    print('Reading subject {}'.format(subject), end='\r')
    avgimage = np.load('{}{}.npy'.format(avg_file, subject))
    maximage = np.load('{}{}.npy'.format(max_file, subject))
    np.save('{}{}{}.npy'.format(outputFile, subject, "1"), maximage)
    np.save('{}{}.npy'.format(outputFile, subject), avgimage)
