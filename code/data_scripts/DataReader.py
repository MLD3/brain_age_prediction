import pandas as pd

def readCSVData(path):
    return pd.read_csv(path)

def readMatrix(path):
    return pd.read_csv(path, header=None).as_matrix()
