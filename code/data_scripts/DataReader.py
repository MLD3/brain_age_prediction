import pandas as pd
from os import listdir
from os.path import join, isfile
import nibabel as nib
import numpy as np
from utils.config import get

def readCSVData(path):
    return pd.read_csv(path)

def readMatrix(path):
    return np.reshape(pd.read_csv(path, header=None).as_matrix(), (get('DATA.MATRICES.DIMENSION'), get('DATA.MATRICES.DIMENSION')))
