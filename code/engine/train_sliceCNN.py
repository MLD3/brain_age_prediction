import tensorflow as tf
import numpy as np
import pandas as pd
import math
import argparse
import os
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from data_scripts.DataPlotter import PlotTrainingValidationLoss
from data_scripts.DataSetNPY import DataSetNPY
from sklearn.model_selection import train_test_split, KFold
from model.build_baselineStructuralCNN import baselineStructuralCNN, sliceCNN
from utils import saveModel
from utils.config import get
from placeholders.shared_placeholders import *
from itertools import product
from engine.trainCommonNPY import *

if __name__ == '__main__':
    PhenotypicsDF = readCSVData(get('DATA.PHENOTYPICS.PATH'))

    BaseDir = get('DATA.SLICES.TOP_DIR')
    xSlicesSuffix = get('DATA.SLICES.X_SLICES_DIR')
    ySlicesSuffix = get('DATA.SLICES.Y_SLICES_DIR')
    zSlicesSuffix = get('DATA.SLICES.Z_SLICES_DIR')

    xSlicesDir = '{}{}'.format(BaseDir, xSlicesSuffix)
    ySlicesDir = '{}{}'.format(BaseDir, ySlicesSuffix)
    zSlicesDir = '{}{}'.format(BaseDir, zSlicesSuffix)

    xSlicesList = [xSlicesSuffix + fileName for fileName in os.listdir(xSlicesDir) if fileName.endswidth('npy')]
    ySlicesList = [ySlicesSuffix + fileName for fileName in os.listdir(ySlicesDir) if fileName.endswidth('npy')]
    zSlicesList = [zSlicesSuffix + fileName for fileName in os.listdir(zSlicesDir) if fileName.endswidth('npy')]

    fileList = np.array(xSlicesList + ySlicesList + zSlicesList)
    labels = np.array(PhenotypicsDF['AgeYears'].tolist())
    dataSet = DataSetNPY(numpyDirectory=StructuralDataDir, numpyFileList=fileList, labels=labels)
