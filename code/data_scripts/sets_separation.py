import numpy as np
import os
import json
import copy
import sys
from sklearn.model_selection import StratifiedKFold
from random import shuffle


def fold_index_helper(i):
    i = i % 10
    return min(i, 9 - i)

def index_helper(i):
    train = [i % 5, (i + 1) % 5, (i + 2) % 5]
    vald = (i + 3) % 5
    test = (i + 4) % 5
    return train, vald, test

def main(inputFile='/data1/brain/PNC_AUGMENTED/combine/', outputFile='/data1/brain/PNC_AUGMENTED/'):
    # inputFile is the source of the original images
    # outputFile is the output location of the set distribution npy file
    if not os.path.exists(outputFile):
        os.makedirs(outputFile)
    train_set, vald_set, test_set = [], [], []
    nameToAgeDict = {}
    _curr_order = np.arange(5)
    np.random.shuffle(_curr_order)
    with open('/data1/brain/UKBIOBANK/ageDistribution.json', 'r') as fp:
        ageDistribution = json.load(fp)
        X, y = [], []
        fold = [[], [], [], [], []]
        for age, filenames in ageDistribution.items():
            for filename in filenames:
                nameToAgeDict[filename] = age
            y.append(age)
        y.sort()
        for age in y:
            X += ageDistribution[age]
        file_index = 0
        # separate the whole dataset into five sets with the same distribution
        for filename in X:
            index = _curr_order[file_index//5]
            fold[index].append(filename)
            file_index += 1
            if file_index // 5 == 0:
                np.random.shuffle(_curr_order)

        for i in range(5):
            train, vald, test = index_helper(i)
            train_set_dummy = np.array([])
            for j in train:
                train_set_dummy = np.concatenate((train_set_dummy, fold[j]))
            train_set += [train_set_dummy]
            vald_set += [fold[vald]]
            test_set += [fold[test]]

        # this is for pseudo generation
        density = [0.25, 0.5, 1, 2, 3]
        for i in range(5):
            shuffle(train_set[i])
            shuffle(vald_set[i])
            shuffle(test_set[i])
            train_set[i] = np.array(train_set[i])
            vald_set[i] = np.array(vald_set[i])
            test_set[i] = np.array(test_set[i])
        train_set, vald_set, test_set = np.array(train_set), np.array(vald_set), np.array(test_set)
        np.save('{}train_list.npy'.format(outputFile), train_set)
        '''
        set_size = 100
        for i in range(5):
            train_set[i] = train_set[i][:set_size]
        np.save('{}combine_train_orig_{}'.format(outputFile, set_size), train_set)
        '''
        '''
        for i in range(5):
            train_set_dummy_dense = []
            for j in range(5):
                cap = train_set[j].shape[0]*density[i]
                train_set_dummy = list(copy.deepcopy(train_set[j]))
                for k in range(1, 6):
                    shuffle(train_set[j])
                    if len(train_set_dummy) - train_set[j].shape[0] > cap:
                        break
                    for image in train_set[j]:
                        if len(train_set_dummy) - train_set[j].shape[0] > cap:
                            break
                        if os.path.exists(inputFile + str(image) + str(k) + '.npy'):
                            age = nameToAgeDict[int(image)]
                            index = ageDistribution[age].index(int(image))
                            combined_image = ageDistribution[age][(index + k) % len(ageDistribution[age])]
                            if str(combined_image) in train_set[j]:
                                train_set_dummy.append(str(image) + str(k))
                train_set_dummy = np.array(train_set_dummy)
                shuffle(train_set_dummy)
                train_set_dummy_dense.append(train_set_dummy)
            train_set_dummy_dense = np.array(train_set_dummy_dense)
            np.save('{}combine_train_{}'.format(outputFile, density[i]), train_set_dummy_dense)
        '''
        np.save('{}vald_list.npy'.format(outputFile), vald_set)
        np.save('{}test_list.npy'.format(outputFile), test_set)
        #print("Successfully generated all the labels")
        print("Successfully generated all the set separations")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])