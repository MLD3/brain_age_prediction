import numpy as np
from utils.args import *
from utils.config import get
from utils.patches import ExtractImagePatches3D, ExtractImagePatchesDEPRECATED
from placeholders.shared_placeholders import *
from model.buildCommon import *

def separableCNN(imagesPL,
                 trainingPL,
                 scale,
                 convolutionalLayers,
                 fullyConnectedLayers,
                 keepProbability=0.6):
    with tf.variable_scope('separableCNN'):
        if imagesPL.dtype != tf.float32:
            imagesPL = tf.cast(imagesPL, tf.float32, name='CastInputToFloat32')
        with tf.variable_scope('PatchExtraction'):
            imagesPL = ExtractImagePatches3D(imagesPL, scale=scale)
        imageOutputs = []
        numChannels = imagesPL.get_shape().as_list()[-1]
        with tf.variable_scope('Convolutions'):
            for channel in range(numChannels):
                with tf.variable_scope('Channel{}'.format(channel)):
                    imageSlice = tf.expand_dims(imagesPL[:, :, :, :, channel], -1)
                    index = 0
                    for numFilters in convolutionalLayers:
                        convStride = (1,1,1)
                        poolStride = [1,2,2,2,1]
                        imageSlice = standardBlock(imageSlice, trainingPL, blockNumber=index,
                                                  filters=numFilters, poolType='MAX', 
                                                  kernelStrides=convStride, 
                                                  poolStrides=poolStride)
                        index += 1
                    imageOutputs.append(imageSlice)
            imagesPL = tf.concat(imageOutputs, axis=-1, name='SliceConcatenation')
        with tf.variable_scope('FullyConnectedLayers'):
            hiddenLayer = tf.layers.flatten(imagesPL)
            for numUnits in fullyConnectedLayers[:-1]:
                hiddenLayer = standardDense(hiddenLayer, units=numUnits, name='hiddenLayer{}'.format(numUnits))
                hiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=keepProbability, is_training=trainingPL)
            outputUnits = fullyConnectedLayers[-1]
            outputLayer = standardDense(hiddenLayer, units=outputUnits, activation=None, use_bias=False, name='outputLayer')
        return outputLayer