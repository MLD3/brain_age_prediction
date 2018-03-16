import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetNPY import DataSetNPY
from utils.saveModel import *
from utils.patches import *
from utils.config import get
from engine.trainCommon import ModelTrainer
from placeholders.shared_placeholders import *

def standardBatchNorm(inputs, trainingPL, momentum=0.9, name=None):
    return tf.layers.batch_normalization(inputs, training=trainingPL, momentum=momentum, name=name, reuse=tf.AUTO_REUSE)

def standardPool(inputs, kernel_size=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name=None):
    return tf.nn.max_pool3d(inputs, ksize=kernel_size, strides=strides, padding=padding, name=name)

def avgPool(inputs, kernel_size=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name=None):
    return tf.nn.avg_pool3d(inputs, ksize=kernel_size, strides=strides, padding=padding, name=name)

def standardConvolution(inputs, filters, kernel_size=(3,3,3), activation=tf.nn.elu, strides=(1,1,1), padding='SAME', name=None):
    return tf.layers.conv3d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation=activation,
                            use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer(), name=name, reuse=tf.AUTO_REUSE)

def standardDense(inputs, units, activation=tf.nn.elu, use_bias=True, name=None):
    if use_bias:
        return tf.layers.dense(inputs=inputs, units=units, activation=activation,
                           bias_initializer=tf.zeros_initializer(),
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           name=name, reuse=tf.AUTO_REUSE)
    else:
        return tf.layers.dense(inputs=inputs, units=units, activation=activation,
                           use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           bias_initializer=tf.zeros_initializer(), name=name, reuse=tf.AUTO_REUSE)

def attentionMap(inputs, randomInit=False):
    with tf.variable_scope('attentionMap'):
        weightShape = inputs.shape.as_list()
        weightShape[0] = 1
        if randomInit:
            attentionWeight = tf.Variable(name='attentionWeight', 
                                              initial_value=tf.random_normal(shape=weightShape, 
                                                   mean=1.0,
                                                   stddev=0.05),
                                              dtype=tf.float32)
        else:
            attentionWeight = tf.Variable(name='attentionWeight',
                                              initial_value=tf.ones(shape=weightShape),
                                              dtype=tf.float32)
        return tf.multiply(inputs, attentionWeight)
    
def standardBlock(inputs,
                  trainingPL,
                  blockNumber,
                  filters,
                  kernelSize=(3,3,3),
                  kernelStrides=(1,1,1),
                  normalize=True,
                  poolStrides=[1,2,2,2,1],
                  poolType='MAX'):
    with tf.variable_scope('ConvBlock{}'.format(blockNumber)):
        BlockConvolution1 = standardConvolution(inputs,
                                                filters=filters,
                                                name='Block{}Convolution1'.format(blockNumber),
                                                kernel_size=kernelSize,
                                                strides=kernelStrides)
        BlockConvolution2 = standardConvolution(BlockConvolution1,
                                                filters=filters,
                                                name='Block{}Convolution2'.format(blockNumber),
                                                kernel_size=kernelSize,
                                                strides=kernelStrides)
        outputLayer = BlockConvolution2

        if normalize:
            outputLayer = standardBatchNorm(outputLayer, trainingPL, name='Block{}BatchNorm'.format(blockNumber))
        if poolType=='MAX':
            outputLayer = standardPool(outputLayer, strides=poolStrides, name='Block{}MaxPool'.format(blockNumber))
        elif poolType=='AVERAGE':
            outputLayer = avgPool(outputLayer, strides=poolStrides, name='Block{}AvgPool'.format(blockNumber))
        return outputLayer

def attentionMapCNN(imagesPL, 
                    trainingPL,
                    strideSize,
                    convFilters,
                    attentionMapBools,
                    keepProbability=0.6,
                    randomAttentionStarter=False):
    convLayers = []
    with tf.variable_scope('ConvolutionalNetwork'):
        if imagesPL.dtype != tf.float32:
            imagesPL = tf.cast(imagesPL, tf.float32, name='CastInputToFloat32')
        if strideSize is not None:
            with tf.variable_scope('PatchExtraction'):
                imagesPL = ExtractImagePatches3D(imagesPL, strideSize=strideSize)
                convLayers.append(imagesPL)
        if randomAttentionStarter:
            attentionCollection = []
            for i in range(8):
                attentionCollection.append(attentionMap(imagesPL, randomInit=True))
            imagesPL = tf.concat(attentionCollection, axis=4)
        
        index = 0
        for numFilters in convFilters:
            if attentionMapBools[index]:
                imagesPL = attentionMap(imagesPL)
            imagesPL = standardBlock(imagesPL, trainingPL, blockNumber=index, filters=numFilters)
            convLayers.append(imagesPL)
            index += 1

        with tf.variable_scope('FullyConnectedLayers'):
            flattenedLayer = tf.layers.flatten(imagesPL)
            hiddenLayer = standardDense(inputs=flattenedLayer, units=256, name='hiddenLayer')
            droppedOutHiddenLayer = tf.contrib.layers.dropout(inputs=hiddenLayer, keep_prob=keepProbability, is_training=trainingPL)
            outputLayer = standardDense(droppedOutHiddenLayer, units=1, activation=None, use_bias=False, name='outputLayer')
        return convLayers
    
    
def restoreAttention():
    additionalArgs = [
        {
        'flag': '--strideSize',
        'help': 'The stride to chunk MRI images into. Typical values are 10, 15, 20, 30, 40, 60.',
        'action': 'store',
        'type': int,
        'dest': 'strideSize',
        'required': True
        },
        {
        'flag': '--type',
        'help': 'One of: traditional, reverse',
        'action': 'store',
        'type': str,
        'dest': 'type',
        'required': True
        },
        {
        'flag': '--attention',
        'help': 'One of: 0, 1, 2, 3',
        'action': 'store',
        'type': int,
        'dest': 'attention',
        'required': True
        }
        ]
    ParseArgs('Run 3D CNN over structural MRI volumes', additionalArgs=additionalArgs, useDefaults=False)
    if GlobalOpts.strideSize <= 0:
        GlobalOpts.strideSize = None
    GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.DOWNSAMPLE_PATH')
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.imageBatchDims = (-1, 61, 73, 61, 1)
    GlobalOpts.trainBatchSize = 4
    GlobalOpts.augment = 'none'
    GlobalOpts.name = 'attention{}_{}_stride{}'.format(GlobalOpts.attention, 
                                                       GlobalOpts.type, 
                                                       GlobalOpts.strideSize)
    modelTrainer = ModelTrainer()
    GlobalOpts.checkpointDir = '{}{}/'.format('../checkpoints/attention_comp/',
                                                     GlobalOpts.name)
    imagesPL, labelsPL = StructuralPlaceholders(GlobalOpts.imageBatchDims)
    trainingPL = TrainingPlaceholder()
        
    if GlobalOpts.type == 'traditional':
        GlobalOpts.convLayers = [8, 16, 32, 64]
    elif GlobalOpts.type == 'reverse':
        GlobalOpts.convLayers = [64, 32, 16, 8]
    
    GlobalOpts.randomAttentionStarter=False
    if GlobalOpts.attention == 0:
        GlobalOpts.attentionMapBools=[True, False, False, False]
    elif GlobalOpts.attention == 1:
        GlobalOpts.attentionMapBools=[False, True, False, False]
    elif GlobalOpts.attention == 2:
        GlobalOpts.attentionMapBools=[True, True, False, False]
    elif GlobalOpts.attention == 3:
        GlobalOpts.attentionMapBools=[False, False, False, False]
        GlobalOpts.randomAttentionStarter=True
    saveFilters(GlobalOpts, imagesPL, trainingPL)

def saveFilters(GlobalOpts, imagesPL, trainingPL):
    convLayers = attentionMapCNN(imagesPL,
                            trainingPL,
                            GlobalOpts.strideSize, 
                            GlobalOpts.convLayers,
                            GlobalOpts.attentionMapBools,
                            randomAttentionStarter=GlobalOpts.randomAttentionStarter)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, GlobalOpts.checkpointDir + GlobalOpts.name + '/run_{}/'.format(0))
        for var in [v for v in tf.global_variables() if 'kernel' in v.name]:
            print(var)
            name = ''.join(ch for ch in var.name if ch.isalnum())
            evalFilter = var.eval()
            np.save(name, evalFilter)
            
def saveOutputLayers(GlobalOpts, imagesPL, trainingPL):
    convLayers = attentionMapCNN(imagesPL,
                            trainingPL,
                            GlobalOpts.strideSize, 
                            GlobalOpts.convLayers,
                            GlobalOpts.attentionMapBools,
                            randomAttentionStarter=GlobalOpts.randomAttentionStarter)
    
    trainDataSet = DataSetNPY(filenames=GlobalOpts.trainFiles,
                                      imageBaseString=GlobalOpts.imageBaseString,
                                      imageBatchDims=GlobalOpts.imageBatchDims,
                                      batchSize=1,
                                      augment=GlobalOpts.augment)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess, GlobalOpts.checkpointDir + GlobalOpts.name + '/run_{}/'.format(0))
        for v in convLayers:
            name = ''.join(ch for ch in v.name if ch.isalnum())
            images, labels = trainDataSet.NextBatch(sess)
            evalLayer = v.eval(feed_dict={
                imagesPL: images,
                trainingPL: False
            })
            print('saving {}'.format(v))
            np.save(name, evalLayer)
        coord.request_stop()
        coord.join(threads)
        
def saveAttentionMaps(GlobalOpts):
    saver = tf.train.Saver()
    attentionMaps = [v for v in tf.global_variables() if 'attentionMap' in v.name]
    
    with tf.Session() as sess:
        for run in range(5):
            saver.restore(sess, GlobalOpts.checkpointDir + GlobalOpts.name + '/run_{}/'.format(run))
            for attentionMap in attentionMaps:
                evalMap = attentionMap.eval()
                np.save('AttentionMap{}'.format(run), evalMap)
                print('Run: {} - Map: {} - Mean: {} - max: {} - min: {} - sd: {}'.format(run, attentionMap.name, np.mean(evalMap), np.max(evalMap), np.min(evalMap), np.std(evalMap)))
                
if __name__ == '__main__':
    restoreAttention()