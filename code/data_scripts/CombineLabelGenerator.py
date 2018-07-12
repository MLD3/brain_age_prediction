import numpy as np
import os
import json
import copy
from sklearn.model_selection import StratifiedKFold
from random import shuffle


def index_helper(i):
    train = [i % 5, (i + 1) % 5, (i + 2) % 5]
    vald = (i + 3) % 5
    test = (i + 4) % 5
    return train, vald, test

def fold_index_helper(i):
    i = i % 10
    return min(i, 9 - i)

'''
# obsolete code: skf will lead to different size of folds
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):
    fold.append(X[test_index])
fold = np.array(fold)
'''

inputFile = '/data1/brain/PNC_AUGMENTED/combine/'
outputFile = '/data1/brain/PNC_AUGMENTED/'
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
train_set, vald_set, test_set = [], [], []
with open('/data1/brain/PNC_AUGMENTED/ageDistribution.json', 'r') as fp:
    ageDistribution = json.load(fp)
    X, y = [], []
    fold = [[], [], [], [], []]
    for age, filenames in ageDistribution.items():
        y.append(age)
    y.sort()
    for age in y:
        X += ageDistribution[age]
    file_index = 0
    for filename in X:
        fold[fold_index_helper(file_index)].append(filename)
        file_index += 1

    for i in range(5):
        train, vald, test = index_helper(i)
        train_set_dummy = np.array([])
        for j in train:
            train_set_dummy = np.concatenate((train_set_dummy, fold[j]))
        train_set += [train_set_dummy]
        vald_set += [fold[vald]]
        test_set += [fold[test]]
    density = [0.25, 0.5, 1, 2, 3]
    for i in range(5):
        train_set[i] = np.array(train_set[i]).astype('<U12')
        vald_set[i] = np.array(vald_set[i]).astype('<U12')
        test_set[i] = np.array(test_set[i]).astype('<U12')
    train_set, vald_set, test_set = np.array(train_set), np.array(vald_set), np.array(test_set)
    np.save('{}train_list.npy'.format(outputFile), train_set)
    for i in range(5):
        train_set_dummy_dense = []
        for j in range(5):
            cap = train_set[j].shape[0]*density[i]
            train_set_dummy = list(copy.deepcopy(train_set[j]))
            for k in range(1, 5):
                shuffle(train_set[j])
                if len(train_set_dummy) - train_set[j].shape[0] > cap:
                    break
                for image in train_set[j]:
                    if len(train_set_dummy) - train_set[j].shape[0] > cap:
                        break
                    if os.path.exists(inputFile + str(image) + str(k) + '.npy'):
                        train_set_dummy.append(str(image) + str(k))
            train_set_dummy = np.array(train_set_dummy).astype('<U12')
            train_set_dummy_dense.append(train_set_dummy)
        train_set_dummy_dense = np.array(train_set_dummy_dense)
        np.save('{}combine_train_{}'.format(outputFile, density[i]), train_set_dummy_dense)
    np.save('{}vald_list.npy'.format(outputFile), vald_set)
    np.save('{}test_list.npy'.format(outputFile), test_set)
    print("Successfully generated all the labels")