import numpy as np
import pandas as pd
from data_scripts.DataReader import *
from data_scripts.DataPlotter import plotHist
from data_scripts.DataSet import DataSet
from utils.config import get, is_file_prefix

class DataHolder(object):
    def __init__(self, df):
        self._df = df
        self.matrices = []
        self.numSubjects = df.shape[0]

    def getBinaryColumn(self, columnName, firstValue, secondValue):
        labels = np.zeros(self._df[columnName].shape)
        labels[self._df[columnName] == secondValue] = 1
        return labels

    def getMatricesFromPath(self, path):
        self.matrices = []
        if path[-1] != '/':
            path += '/'
        for subjectID in self._df['Subject']:
            matrixPath = path + str(subjectID) + '.csv'
            matrix = readMatrix(matrixPath)
            self.matrices.append(matrix)

    def matricesToImages(self):
        for index in range(self.numSubjects):
            np.fill_diagonal(self.matrices[index], 0)

    def returnDataSet(self):
        mats = np.array(self.matrices)
        mats = np.reshape(mats, (mats.shape[0], mats.shape[1], mats.shape[2], 1))
        labels = np.array(self._df['AgeYears'].values.copy())
        labels = np.reshape(labels, (labels.shape[0], 1))
        return DataSet(mats, labels, reshape=True)

    def filterByColumn(self, columnName, equalValue):
        return self._df[self._df[columnName] == equalValue]

    def getNumberByFilteredColumn(self, columnName, equalValue):
        return self.filterByColumn(columnName, equalValue).shape[0]

    def getMean(self, onColumn, filterColumn='', equalValue=''):
        if filterColumn != '' and equalValue != '':
            return np.mean(self.filterByColumn(filterColumn, equalValue)[onColumn])
        else:
            return np.mean(self._df[onColumn])

    def getSD(self, onColumn, filterColumn='', equalValue=''):
        if filterColumn != '' and equalValue != '':
            return np.std(self.filterByColumn(filterColumn, equalValue)[onColumn])
        else:
            return np.std(self._df[onColumn])

if __name__ == '__main__':
    dataHolder = DataHolder(readCSVData(get('DATA.PHENOTYPICS.PATH')))
    #dataHolder.getMatricesFromPath(get('DATA.MATRICES.PATH'))
    print("Number of subjects: " + '%f' % dataHolder.numSubjects)
    print("Age mean: " + '%f' % dataHolder.getMean('AgeYears') + " Age SD: " + '%f' % dataHolder.getSD('AgeYears'))
    print("Scrub mean: " + '%f' % dataHolder.getMean('ScrubRatio') + " Scrub SD: " + '%f' % dataHolder.getSD('ScrubRatio'))
    print("FD mean: " + '%f' % dataHolder.getMean('meanFD') + " FD SD: " + '%f' % dataHolder.getSD('meanFD'))
    print("Number of female subjects: " + '%f' % dataHolder.getNumberByFilteredColumn('Sex', 'F'))
    print("Female age mean: " + '%f' % dataHolder.getMean('AgeYears', 'Sex', 'F') + " Female age SD: " + '%f' % dataHolder.getSD('AgeYears', 'Sex', 'F'))
    print("Female scrub mean: " + '%f' % dataHolder.getMean('ScrubRatio', 'Sex', 'F') + " Female scrub SD: " + '%f' % dataHolder.getSD('ScrubRatio', 'Sex', 'F'))
    print("Female FD mean: " + '%f' % dataHolder.getMean('meanFD', 'Sex', 'F') + " Female FD SD: " + '%f' % dataHolder.getSD('meanFD', 'Sex', 'F'))
    print("Number of male subjects: " + '%f' % dataHolder.getNumberByFilteredColumn('Sex', 'M'))
    print("Male age mean: " + '%f' % dataHolder.getMean('AgeYears', 'Sex', 'M') + " Male age SD: " + '%f' % dataHolder.getSD('AgeYears', 'Sex', 'M'))
    print("Male scrub mean: " + '%f' % dataHolder.getMean('ScrubRatio', 'Sex', 'M') + " Male scrub SD: " + '%f' % dataHolder.getSD('ScrubRatio', 'Sex', 'M'))
    print("Male FD mean: " + '%f' % dataHolder.getMean('meanFD', 'Sex', 'M') + " Male FD SD: " + '%f' % dataHolder.getSD('meanFD', 'Sex', 'M'))

    plotHist(dataHolder._df['AgeYears'], saveName='AgeHist.png', title='Age counts of all subjects')
    plotHist(dataHolder._df['meanFD'], saveName='FDHist.png', title='Mean Frame Displacement (FD) rate of all subjects')
    plotHist(dataHolder._df['ScrubRatio'], saveName='ScrubHist.png', title='Scrub Ratio counts of all subjects')

    plotHist(dataHolder.filterByColumn('Sex', 'F')['AgeYears'], saveName='FemaleAgeHist.png', title='Age counts of female subjects')
    plotHist(dataHolder.filterByColumn('Sex', 'F')['meanFD'], saveName='FemaleFDHist.png', title='Mean Frame Displacement (FD) rate of female subjects')
    plotHist(dataHolder.filterByColumn('Sex', 'F')['ScrubRatio'], saveName='FemaleScrubHist.png', title='Scrub Ratio counts of female subjects')

    plotHist(dataHolder.filterByColumn('Sex', 'M')['AgeYears'], saveName='MaleAgeHist.png', title='Age counts of male subjects')
    plotHist(dataHolder.filterByColumn('Sex', 'M')['meanFD'], saveName='MaleFDHist.png', title='Mean Frame Displacement (FD) rate of male subjects')
    plotHist(dataHolder.filterByColumn('Sex', 'M')['ScrubRatio'], saveName='MaleScrubHist.png', title='Scrub Ratio counts of male subjects')
