import numpy as np
import os
import json
import copy
from sklearn.model_selection import StratifiedKFold
from random import shuffle


inputFile = '/data1/brain/PNC_AUGMENTED/'
outputFile = '/data1/brain/PNC_AUGMENTED/'
for size in [100, 200]:
    augmented = inputFile + 'combine_train_1_{}.npy'.format(size)
    augmentedList = np.load(augmented)
    orig_list = []
    for i in range(5):
        orig_list_item = [filename for filename in augmentedList[i] if len(filename) == 12]
        orig_list_item = np.array(orig_list_item)
        orig_list.append(orig_list_item)
    np.save(outputFile + 'combine_train_orig_{}.npy'.format(size), orig_list)