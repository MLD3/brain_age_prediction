import pandas as pd
from os import listdir
from os.path import join, isfile
import nibabel as nib
import numpy as np

def readCSVData(path):
    return pd.read_csv(path)

def readMatrix(path):
    return pd.read_csv(path, header=None).as_matrix()

def readNIIData(path):
	filenames = []
	for f in listdir(path):
		if f.endswith(".nii"):
			filenames.append(join(path, f))
	# find shape of one images
	num_subjects = len(filenames)
	instance = nib.load(filenames[0])

	data = np.zeros((num_subjects, instance.shape[0], instance.shape[1], instance.shape[2], instance.shape[3]))
	for i in range(num_subjects):
		instance = nib.load(filenames[i])
		data[i, :, :, :, :] = instance.get_data()
	return data
