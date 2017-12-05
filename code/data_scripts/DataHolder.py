import numpy as np
import pandas as pd
from data_scripts.DataReader import *
from data_scripts.DataPlotter import plotHist
from data_scripts.DataSet import DataSet
from utils.config import get, is_file_prefix
import nibabel as nib
from sklearn.model_selection import train_test_split
import os


class DataHolder(object):
    def __init__(self, df):
        self._df = df
        self.matrices = []
        self.numSubjects = df.shape[0]
        self.train_images = []
        self.test_images = []
        self.train_subject = []
        self.test_subject = []

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

    def get_independent_image(self, image, train = True):
        if train:
            for i in range(120):
                self.train_images.append(image.get_data()[:,:,:,i])
        else:
            for i in range(120):
                self.test_images.append(image.get_data()[:,:,:,i])

    def getNIIImagesFromPath(self, path):
        self.matrices = []
        if path[-1] != '/':
            path += '/'

        subjects = []
        for subject_id in self._df['Subject']:
            subjects.append(int(subject_id))
        print("Subjects")
        print(len(subjects))
        self.train_subject, self.test_subject = train_test_split(subjects, test_size = 0.2)
        print("Number of subjects in train and test")
        print(len(self.train_subject))
        print(len(self.test_subject))
        for subject_id in self.train_subject:
            image_path = path + "s6_" + str(subject_id) + ".nii"
            if os.path.isfile(image_path):
                image = nib.load(image_path)
                self.get_independent_image(image, train = True)
            else:
                print("Train image not exists" + str(subject_id))
        print("number of images in train")
        print(len(self.train_images))
        for subject_id in self.test_subject:
            image_path = path + "s6_" + str(subject_id) + ".nii"
            if os.path.isfile(image_path):
                image = nib.load(image_path)
                self.get_independent_image(image, train = False)
                # self.matrices.append(image.get_data())
        print("number of images in test")
        print(len(self.test_images))

        
    def copy_labels(self, labels):
        copied_label = np.zeros((labels.shape[0] * 120, 1))
        for i in range(labels.shape[0]):
            for j in range(120):
                copied_label[120 * i + j, 0] = labels[i]
        return copied_label

    def returnNIIDataset(self):
        train_mats = np.array(self.train_images)
        print(train_mats.shape)
        train_mats = np.reshape(train_mats, (train_mats.shape[0], train_mats.shape[1], train_mats.shape[2], train_mats.shape[3], 1))
        train_labels = np.zeros((len(self.train_subject)))
        for idx in range(len(self.train_subject)):
            train_labels[idx] = self._df.loc[self._df['Subject'] == self.train_subject[idx]]['AgeYears']
        # labels = np.array(self._df['AgeYears'].values.copy())
        train_labels = self.copy_labels(train_labels)
        print(train_labels.shape)
        print(train_mats.shape)
        test_mats = np.array(self.test_images)
        test_mats = np.reshape(test_mats, (test_mats.shape[0], test_mats.shape[1], test_mats.shape[2], test_mats.shape[3], 1))
        test_labels = np.zeros((len(self.test_subject)))
        for idx in range(len(self.test_subject)):
            test_labels[idx] = self._df.loc[self._df['Subject'] == self.test_subject[idx]]['AgeYears']
        test_labels = self.copy_labels(test_labels)
        print(test_labels.shape)
        print(test_mats.shape)
        return DataSet(train_mats, train_labels, reshape=True, fMRI=True), DataSet(test_mats, test_labels, reshape=True, fMRI=True)


    def matricesToImages(self):
        for index in range(self.numSubjects):
            np.fill_diagonal(self.matrices[index], 1)
            self.matrices[index] = np.reshape(self.matrices[index], (get('DATA.MATRICES.DIMENSION'), get('DATA.MATRICES.DIMENSION'), 1))

    def returnDataSet(self):
        mats = np.array(self.matrices)
        mats = np.reshape(mats, (mats.shape[0], mats.shape[1], mats.shape[2], 1))
        labels = np.array(self._df['AgeYears'].values.copy())
        labels = np.reshape(labels, (labels.shape[0], 1))
        return DataSet(mats, labels)

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
