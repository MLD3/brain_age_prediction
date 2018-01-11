import tensorflow as tf
import numpy as np
import pandas as pd
import math
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from data_scripts.DataPlotter import PlotTrainingValidationLoss
from data_scripts.DataSetNPY import DataSetNPY
from sklearn.model_selection import train_test_split, KFold
from model.build_baselineROICNN import baselineROICNN
from utils import saveModel
from utils.config import get
from placeholders.shared_placeholders import *
from itertools import product
from engine.trainCommon import *

if __name__ == '__main__':
    PhenotypicsDF = readCSVData(get('DATA.PHENOTYPICS.PATH'))
    StructuralDataDir = get('DATA.STRUCTURAL.NUMPY_PATH')
    fileList = np.array([subject + '.npy' for subject in PhenotypicsDF['Subject'].tolist()])
    labels = np.array(PhenotypicsDF['AgeYears'].tolist())
    dataSet = DataSetNPY(numpyDirectory=StructuralDataDir, numpyFileList=fileList, labels=labels)
