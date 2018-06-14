import pandas as pd
import numpy as np
from skimage.measure import block_reduce

inputFile = 'PNC/labels/'
outputFile = 'PNC_AUGMENTED/pool_mix_labels/'

train_file = np.load('PNC/train.npy').tolist()
new_train_file = train_file[:]
for filename in train_file:
    new_train_file.append(filename + '1')
np.random.shuffle(new_train_file)
np.save('PNC_AUGMENTED/concat_train_list.npy', new_train_file)
'''
new_train_file = train_file[:]
for filename in train_file:
    new_train_file.append(filename + '1')
    new_train_file.append(filename + '2')
np.random.shuffle(new_train_file)
np.save('PNC_AUGMENTED/pool_mix_train_list_2.npy', new_train_file)
new_train_file = train_file[:]
for filename in train_file:
    new_train_file.append(filename + '1')
    new_train_file.append(filename + '2')
    new_train_file.append(filename + '3')
np.random.shuffle(new_train_file)
np.save('PNC_AUGMENTED/pool_mix_train_list_3.npy', new_train_file)
new_train_file = train_file[:]
for filename in train_file:
    new_train_file.append(filename + '1')
    new_train_file.append(filename + '2')
    new_train_file.append(filename + '3')
    new_train_file.append(filename + '4')
np.random.shuffle(new_train_file)
np.save('PNC_AUGMENTED/pool_mix_train_list_4.npy', new_train_file)
'''