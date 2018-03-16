import tensorflow as tf
import numpy as np
from utils.config import get
from utils.patches import ExtractImagePatches3D
from placeholders.shared_placeholders import *
from buildCommon import *

def customCNN(imagesPL,
              trainingPL,
              scale,
              convolutionalLayers,
              fullyConnectedLayers,
              keepProbability=0.6,
              convStrides=None,
              poolStrides=None,
              poolType='MAX'):
    with tf.variable_scope('customCNN'):
        if imagesPL.dtype != tf.float32:
            imagesPL = tf.cast(imagesPL, tf.float32, name='CastInputToFloat32')
        if strideSize is not None:
            with tf.variable_scope('PatchExtraction'):
                imagesPL = ExtractImagePatches3D(imagesPL, scale=scale)
        index = 0
        for numFilters in convolutionalLayers:
            convStride = (1,1,1)
            if convStrides is not None:
                convStride = (convStrides[index],) * 3
            poolStride = [1,2,2,2,1]
            if poolStrides is not None:
                poolStride = [1, poolStrides[index], poolStrides[index], poolStrides[index], 1]

            imagesPL = standardBlock(imagesPL, trainingPL, blockNumber=index, filters=numFilters, poolType=poolType, kernelStrides=convStride, poolStrides=poolStride)
            index += 1
        with tf.variable_scope('FullyConnectedLayers'):
            hiddenLayer = tf.layers.flatten(imagesPL)
            for numUnits in fullyConnectedLayers[:-1]:
                hiddenLayer = standardDense(hiddenLayer, units=numUnits, name='hiddenLayer{}'.format(numUnits))
                hiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=keepProbability, is_training=trainingPL)
        outputUnits = fullyConnectedLayers[-1]
        outputLayer = standardDense(hiddenLayer, units=outputUnits, activation=None, use_bias=False)
        return outputLayer
