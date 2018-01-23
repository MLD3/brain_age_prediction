import os
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

from utils.config import get

class DataSetNPY(object):
    def __init__(self, binFile):
        self.binFile = binFile
        self.numExamplesPerEpoch
        
