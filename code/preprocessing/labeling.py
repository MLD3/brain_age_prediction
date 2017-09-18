import numpy as np

from preprocessing import preprocess
from preprocessing import augmentation
from data_scripts.DataSet import DataSet

def writeBlankLabels(outFile='../data/train/labels.txt', imageDir='../data/train/cropped', imageExtension='.png'):
    images, names = preprocess.getAllImagesInDirectory(dir=imageDir, imageExtension=imageExtension)
    labels = np.zeros((len(images)), dtype=int)

    f = open(outFile, 'w')
    for index in range(len(names)):
        f.write(names[index] + ',' + str(labels[index]) + '\n')

    f.close()

def getImagesAndLabels(labelsFile='../data/train/labels.txt', imageDir='../data/train/cropped', imageExtension='.png'):
    images, names = preprocess.getAllImagesInDirectory(dir=imageDir, imageExtension=imageExtension)
    labels = []

    f = open(labelsFile, 'r')
    counter = 0
    for line in f:
        label = int(line.split(',')[-1])
        labels.append(label)

        imageName = line.split(',')[0]
        if (names[counter] != imageName):
            raise ValueError('Names' + names[counter] + ',' + imageName +
            ' in the label file and in the image directory do not match.')
        counter += 1

    return (images, names, np.array(labels))

def getAugmentedData(labelsFile='../data/train/labels.txt', imageDir='../data/train/cropped', imageExtension='.png'):
    images, names, labels = getImagesAndLabels(labelsFile, imageDir, imageExtension)
    noFaceIndices = labels == 0
    noFaceImages = images[noFaceIndices]
    noFaceLabels = labels[noFaceIndices]

    faceIndices = labels == 1
    faceImages = images[faceIndices]
    faceLabels = labels[faceIndices]

    augmentedFaceImages = []
    augmentedFaceLabels = []
    for index in range(np.sum(faceIndices)):
        faceImage = faceImages[index]
        augmentedSet = augmentation.augmentImage(faceImage)
        for augmentedImage in augmentedSet:
            augmentedFaceImages.append(augmentedImage)
            augmentedFaceLabels.append(1)

    augmentedFaceImages = np.array(augmentedFaceImages)
    augmentedFaceLabels = np.array(augmentedFaceLabels)

    noFaceImagesDownsampled = []
    for image in noFaceImages:
        noFaceImagesDownsampled.append(preprocess.downsample(image, (128, 128)))
    noFaceImagesDownsampled = np.array(noFaceImagesDownsampled)

    faceImagesDownsampled = []
    for image in augmentedFaceImages:
        faceImagesDownsampled.append(preprocess.downsample(image, (128, 128)))
    faceImagesDownsampled = np.array(faceImagesDownsampled)

    return (noFaceImagesDownsampled, noFaceLabels, faceImagesDownsampled, augmentedFaceLabels)

def getAugmentedDataSet(labelsFile='../data/train/labels.txt', imageDir='../data/train/cropped', imageExtension='.png', oneHot=True):
    (noFaceImagesDownsampled, noFaceLabels, faceImagesDownsampled, augmentedFaceLabels) = getAugmentedData(labelsFile=labelsFile, imageDir=imageDir, imageExtension=imageExtension)

    images = np.append(noFaceImagesDownsampled, faceImagesDownsampled, axis=0)
    labels = np.append(noFaceLabels, augmentedFaceLabels, axis=0)
    dataset = DataSet(images=images, labels=labels, oneHot=oneHot)

    return dataset

def getData(labelsFile='../data/test/labels.txt', imageDir='../data/test/cropped', imageExtension='.png'):
    images, names, labels = getImagesAndLabels(labelsFile, imageDir, imageExtension)
    noFaceIndices = labels == 0
    noFaceImages = images[noFaceIndices]
    noFaceLabels = labels[noFaceIndices]

    faceIndices = labels == 1
    faceImages = images[faceIndices]
    faceLabels = labels[faceIndices]

    noFaceImagesDownsampled = []
    for image in noFaceImages:
        noFaceImagesDownsampled.append(preprocess.downsample(image, (128, 128)))
    noFaceImagesDownsampled = np.array(noFaceImagesDownsampled)

    faceImagesDownsampled = []
    for image in faceImages:
        faceImagesDownsampled.append(preprocess.downsample(image, (128, 128)))
    faceImagesDownsampled = np.array(faceImagesDownsampled)

    return (noFaceImagesDownsampled, noFaceLabels, faceImagesDownsampled, faceLabels)

def getDataSet(labelsFile='../data/train/labels.txt', imageDir='../data/train/cropped', imageExtension='.png', oneHot=True):
    (noFaceImagesDownsampled, noFaceLabels, faceImagesDownsampled, faceLabels) = getData(labelsFile=labelsFile, imageDir=imageDir, imageExtension=imageExtension)
    images = np.append(noFaceImagesDownsampled, faceImagesDownsampled, axis=0)

    labels = np.append(noFaceLabels, faceLabels, axis=0)
    dataset = DataSet(images=images, labels=labels, oneHot=oneHot)

    return dataset


if __name__ == '__main__':
    print("Nothing to be done here!")
