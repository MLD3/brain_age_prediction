import numpy as np
import os
import json

inputFile = '/data1/brain/PNC/structural/avgpool3x3x3/'
outputFile = '/data1/brain/PNC_AUGMENTED/combine/'
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
with open('/data1/brain/PNC_AUGMENTED/ageDistribution.json', 'r') as fp:
    ageDistribution = json.load(fp)
    for age, files in ageDistribution.items():
        if len(files) == 1:
            continue
        images = []
        for file in files:
            image = np.load('{}{}.npy'.format(inputFile, file))
            np.save('{}{}.npy'.format(outputFile, file))
            images.append(image)
        for i in range(len(files)):
            for j in range(1, min(len(files)-1, 6)):
                k = (i + j) % len(files)
                newimage = np.concatenate((images[i][0:21], images[k][21:41]))
                assert(newimage.shape == (41, 49, 41))
                np.save('{}{}{}.npy'.format(outputFile, files[i], j), newimage)
    '''
    ## These lines are for checking the original set distribution
    ageList = []
    for age, files in ageDistribution.items():
        ageList.append(age)
    ageList.sort()
    for age in ageList:
        print("Age: " + str(int(age)/12) + " with " + str(len(ageDistribution[age])) + " files")
    '''