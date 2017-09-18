import numpy as np
import pandas as pd

class DataReader(object):
    def __init__(self, path):
        self._path = path

    def readCSVData(self, path=""):
        if path != "":
            self._path = path

        df = pd.read_csv(self._path)
        return df

if __name__ == '__main__':
