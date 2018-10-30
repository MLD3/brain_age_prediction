import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import pandas as pd


def viewAgeDistribution(labelsloc="/data1/brain/PNC/labels/", group=None, save=False, index=None):
    ageDistDict = {}
    ageList = []
    ageDist = []
    ageEnum = []
    if not labelsloc:
        labelsloc = "/data1/brain/PNC/labels/"
    if not group:
        df = pd.read_csv('/data1/brain/PNC/PNC_724_phenotypics.csv')
        for _, row in df.iterrows():
            subject = row['Subject']
            age = np.load('{}{}{}'.format(labelsloc, subject, '.npy'))
            ageEnum.append(age[0])        
            age = int(age[0])
            if age not in ageDistDict:
                ageDistDict[age] = []
            ageDistDict[age].append(subject)
    else:
        df = np.load(group)
        if index is not None:
            df = df[index]
        for subject in np.nditer(df):
            age = np.load('{}{}{}'.format(labelsloc, subject, '.npy'))
            ageEnum.append(age[0]) 
            age = int(age[0])
            if age not in ageDistDict:
                ageDistDict[age] = []
            ageDistDict[age].append(subject)
    for key, value in ageDistDict.items():
        ageList.append(key)
    ageList.sort()
    for age in ageList:
        ageDist.append(len(ageDistDict[age]))
    ind = np.arange(len(ageList))
    width = 0.5 # arbitrarily chosen const
    ageEnum = np.array(ageEnum)
    print("The average age is " + str(np.mean(ageEnum)) + " with a std of " + str(np.std(ageEnum)) )
    print("The length of the set is :" + str(ageEnum.shape[0]))
    fig, ax = plt.subplots()
    rect = ax.bar(ind, ageDist, color='r')
    ax.set_ylabel('Number of MRI in this age')
    ax.set_title('MRI distribution by the age')
    ax.set_xticks(ind)
    ax.set_xticklabels(ageList)
    plt.show()
    if save:
        with open('/data1/brain/PNC_AUGMENTED/ageDistribution.json', 'w') as fp:
            json.dump(ageDistDict, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the age distribution in the data.')
    parser.add_argument('--loc', help='location of the labels')
    parser.add_argument('--group', help='location of the group identifier npy file')
    parser.add_argument('--save', help='whether to save or not', action='store_true')
    parser.add_argument('--index', help='index of the train group', action='store', type=int)
    args = parser.parse_args()
    viewAgeDistribution(args.loc, args.group, args.save, args.index)