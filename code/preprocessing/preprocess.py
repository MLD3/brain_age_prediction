import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imsave, imresize
from scipy import ndimage
from os import listdir

def saveImage(name, image):
    imsave(name, image)

def getImage(fromPath):
    return ndimage.imread(fromPath)

def getImageNamesInDirectory(dir='../data/train/raw', imageExtension=".jpg"):
    return [image for image in listdir(dir) if image.endswith(imageExtension)]

def getAllImagesInDirectory(dir='../data/train/raw', imageExtension=".jpg"):
    imageNames = getImageNamesInDirectory(dir, imageExtension)
    return (np.array([getImage(dir + '/' + imageName) for imageName in imageNames]), imageNames)

def getMinimumCropDimension(images):
    shapes = [im.shape for im in images]
    heights = [s[0] for s in shapes]
    widths = [s[1] for s in shapes]
    return min(min(widths), min(heights))

def segmentImage(image, dimension):
    height, width, depth = image.shape
    currentX = dimension
    currentY = dimension
    imageChunks = []
    while currentY <= height:
        while currentX <= width:
            imageChunk = image[(currentY-dimension):currentY,(currentX-dimension):currentX,:]
            imageChunks.append(imageChunk)
            currentX += dimension
        currentX = dimension
        currentY += dimension
    return np.array(imageChunks) # (numChunks, dimension, dimension, depth)

def cropAndReplicateImagesToMinimumDimension(images, names, imageExtension='.jpg'):
    if len(images) != len(names):
        raise ValueError("The number of names must match the number of images.")

    minDimension = getMinimumCropDimension(images)
    croppedImages = []
    croppedNames = []
    for index in range(len(images)):
        imageChunks = segmentImage(images[index], minDimension)
        name = names[index].split('.')[0]
        counter = 0
        for imageChunk in imageChunks:
            croppedImages.append(imageChunk)
            croppedName = name + '_chunk' + str(counter) + imageExtension
            croppedNames.append(croppedName)
            counter += 1

    return (np.array(croppedImages), croppedNames)

def cropImagesFromDirectory(dir='../data/train/raw', imageExtension='.jpg', outDir='../data/train/cropped'):
    images, names = getAllImagesInDirectory(dir=dir, imageExtension=imageExtension)
    croppedImages, croppedNames = cropAndReplicateImagesToMinimumDimension(images, names, imageExtension)
    for index in range(len(croppedImages)):
        saveImage(outDir + '/' + croppedNames[index], croppedImages[index])

def transferImagesFormat(dir='../data/train/croppedjpg', informat='.jpg', outdir='../data/train/cropped', outformat='.png'):
    images, names = getAllImagesInDirectory(dir=dir, imageExtension=informat)
    for index in range(len(images)):
        name = names[index].split('.')[0]
        saveImage(outdir + '/' + name + outformat, images[index])

def downsample(image, newSize, interp='bicubic'):
    return imresize(image, size=newSize, interp=interp)

def showImage(image):
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    print("Nothing to do here!")
