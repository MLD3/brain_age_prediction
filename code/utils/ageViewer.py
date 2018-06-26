import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import pandas as pd


def viewAgeDistribution(labelsloc="/data1/brain/PNC/labels/"):
    ageDistDict = {}
    ageList = []
    ageDist = []
    df = pd.read_csv('PNC/PNC_724_phenotypics.csv')
    for _, row in df.iterrows():
        subject = row['Subject']
        print('Reading subject {}'.format(subject), end='\r')
        age = np.load('{}{}'.format(labelsloc, subject))
        if age not in ageDistDict:
            ageDistDict[age] = []
        ageDistDict[age].append(subject)
    for key, value in ageDistDict.items():
        ageList.append(key)
        ageDist.append(len(value))
    ind = np.arange(len(ageList))
    width = 0.5 # arbitrarily chosen const
    fig, ax = plt.subplots()
    rect = ax.bar(ind, ageDist, width, color='r')
    ax.set_ylabel('Number of MRI in this age')
    ax.set_title('MRI distribution by the age')
    ax.set_xticks(ind)
    ax.set_xticklabels(ageList)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='Plot the age distribution in the data.')
    parser.add_argument('--loc', help='location of the age distribution')
    '''
    viewAgeDistribution()