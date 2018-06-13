import pandas as pd
import numpy as np
from skimage.measure import block_reduce

inputFile = 'PNC/structural/numpyArrays/'
outputFile = 'PNC_AUGMENTED/pool_mix/'

df = pd.read_csv('PNC/PNC_724_phenotypics.csv')

def mix(a, axis=None, out=None):
    avg_pool_rate = 0.75
    if np.random.random_sample() >= avg_pool_rate:
        return np.max(a, axis, out)
    else:
        return np.mean(a, axis, out)

for _, row in df.iterrows():
    subject = row['Subject']
    print('Reading subject {}'.format(subject), end='\r')
    image = np.load('{}{}.npy'.format(inputFile, subject))
    mix_image1 = block_reduce(image, block_size=(3,3,3), func=mix, cval=0.0)
    mix_image2 = block_reduce(image, block_size=(3,3,3), func=mix, cval=0.0)
    mix_image3 = block_reduce(image, block_size=(3,3,3), func=mix, cval=0.0)
    mix_image4 = block_reduce(image, block_size=(3,3,3), func=mix, cval=0.0)
    original = block_reduce(image, block_size=(3,3,3), func=np.mean, cval=0.0)
    np.save('{}{}{}.npy'.format(outputFile, subject, "1"), mix_image1)
    np.save('{}{}{}.npy'.format(outputFile, subject, "2"), mix_image2)
    np.save('{}{}{}.npy'.format(outputFile, subject, "3"), mix_image3)
    np.save('{}{}{}.npy'.format(outputFile, subject, "4"), mix_image4)
    np.save('{}{}.npy'.format(outputFile, subject), original)
