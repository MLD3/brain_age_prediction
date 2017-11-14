import numpy as np
import pandas as pd
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from utils.config import get, is_file_prefix

def MSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

if __name__ == '__main__':
    dataHolder = DataHolder(readCSVData(get('DATA.PHENOTYPICS.PATH')))
    dataHolder.getMatricesFromPath(get('DATA.MATRICES.PATH'))
    X = np.array([mat[np.tril_indices(mat.shape[0], k=-1)] for mat in dataHolder.matrices])
    Y = dataHolder._df['AgeYears'].values.copy()

    averageY = np.mean(Y)
    y_pred = np.array([averageY] * len(Y))
    loss = MSE(Y, y_pred)

    print("Baseline: Using the average age as a prediction")
    print("Evaluated MSE: %f" % loss)
