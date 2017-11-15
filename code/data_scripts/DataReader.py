import pandas as pd
from utils import get

def readCSVData(path):
    return pd.read_csv(path)

def readMatrix(path):
    return np.reshape(pd.read_csv(path, header=None).as_matrix(), (get('DATA.MATRICES.DIMENSION'), get('DATA.MATRICES.DIMENSION'), 1))
