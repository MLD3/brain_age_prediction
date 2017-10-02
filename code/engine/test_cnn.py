import tensorflow as tf
import numpy as np
import pandas as pd
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from utils.config import get, is_file_prefix
from data_scripts.DataSet import DataSet
from model.build_cnn import *



if __name__ == '__main__':
    with tf.train.MonitoredTrainingSession as monitoredSess:
        
