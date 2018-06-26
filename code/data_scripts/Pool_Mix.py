import pandas as pd
import numpy as np
from skimage.measure import block_reduce
import os

inputFile = 'PNC/structural/maxpoolArrays/'
df = pd.read_csv('PNC/PNC_724_phenotypics.csv')
max_pool_rate = 0.25
def mix(a, axis=None, out=None):
    # avg_pool_rate = 0.75
    if np.random.random_sample() <= max_pool_rate:
        return np.max(a, axis, out)
    else:
        return np.mean(a, axis, out)
for max_pool_rate_index in range(1,20):
    max_pool_rate = max_pool_rate_index * 0.05
    outputFile = 'PNC_AUGMENTED/pool_mix_{}/'.format(max_pool_rate)
    if not os.path.exists(outputFile):
        os.makedirs(outputFile)
    for _, row in df.iterrows():
        subject = row['Subject']
        print('Reading subject {}'.format(subject), end='\r')
        image = np.load('{}{}.npy'.format(inputFile, subject))
        '''
        mix_image1 = block_reduce(image, block_size=(3,3,3), func=mix, cval=0.0)
        mix_image2 = block_reduce(image, block_size=(3,3,3), func=mix, cval=0.0)
        mix_image3 = block_reduce(image, block_size=(3,3,3), func=mix, cval=0.0)
        mix_image4 = block_reduce(image, block_size=(3,3,3), func=mix, cval=0.0)
        original = block_reduce(image, block_size=(3,3,3), func=np.mean, cval=0.0)
        np.save('{}{}{}.npy'.format(outputFile, subject, "1"), mix_image1)
        np.save('{}{}{}.npy'.format(outputFile, subject, "2"), mix_image2)
        np.save('{}{}{}.npy'.format(outputFile, subject, "3"), mix_image3)
        np.save('{}{}{}.npy'.format(outputFile, subject, "4"), mix_image4)
        '''
        np.save('{}{}{}.npy'.format(outputFile, subject, "5"), image)
