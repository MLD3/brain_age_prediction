import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from data_scripts.DataSetNPY import DataSetNPY
from model.buildCustomCNN import customCNN
from model.buildSeparableCNN import separableCNN
from utils.saveModel import *
from utils.config import get
from engine.trainCommon import *
from placeholders.shared_placeholders import *

def GetTrainingOperation(lossOp, learningRate):
    """
    Given a loss operation and a learning rate,
    returns an operation to minimize that loss and
    the corresponding gradients.
    """
    with tf.variable_scope('optimizer'):
        if GlobalOpts.regStrength is not None:
            regularizerLosses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            lossOp = tf.add_n([lossOp] + regularizerLosses, name="RegularizedLoss")
        updateOp, gradients = AdamOptimizer(lossOp, learningRate)
    return updateOp, gradients

def GetDataSetInputs():
    """
    Returns dataset objects based on the specified input parameters.
    """
    with tf.variable_scope('Inputs'):
        with tf.variable_scope('TrainingInputs'):
            trainDataSets = []
            for i in range(5):
                trainDataSet = DataSetNPY(filenames=GlobalOpts.trainFiles[i],
                                          imageBaseString=GlobalOpts.imageBaseString,
                                          imageBatchDims=GlobalOpts.imageBatchDims,
                                          labelBaseString=GlobalOpts.labelBaseString,
                                          batchSize=GlobalOpts.batchSize,
                                          augment=GlobalOpts.augment,
                                          augRatio=GlobalOpts.augRatio)
                trainDataSets.append(trainDataSet)
        with tf.variable_scope('ValidationInputs'):
            valdDataSets = []
            for i in range(5):
                valdDataSet  = DataSetNPY(filenames=GlobalOpts.valdFiles[i],
                                         imageBaseString=GlobalOpts.imageBaseString,
                                         imageBatchDims=GlobalOpts.imageBatchDims,
                                         labelBaseString=GlobalOpts.labelBaseString,
                                         batchSize=1,
                                         maxItemsInQueue=GlobalOpts.numberValdItems[i],
                                         shuffle=False)
                valdDataSets.append(valdDataSet)
        with tf.variable_scope('TestInputs'):
            testDataSets = []
            for i in range(5):
                testDataSet  = DataSetNPY(filenames=GlobalOpts.testFiles[i],
                                         imageBaseString=GlobalOpts.imageBaseString,
                                         imageBatchDims=GlobalOpts.imageBatchDims,
                                         labelBaseString=GlobalOpts.labelBaseString,
                                         batchSize=1,
                                         maxItemsInQueue=GlobalOpts.numberTestItems[i],
                                         shuffle=False)
                testDataSets.append(testDataSet)
    return trainDataSets, valdDataSets, testDataSets

def DefineDataOpts(data='PNC', summaryName='test_comp'):
    """
    Defines global parameters based on input parameters.
    """
    if GlobalOpts.dataScale == 1:
        GlobalOpts.imageBatchDims = (-1, 121, 145, 121, 1)
    elif GlobalOpts.dataScale == 2:
        GlobalOpts.imageBatchDims = (-1, 61, 73, 61, 1)
    elif GlobalOpts.dataScale == 3:
        GlobalOpts.imageBatchDims = (-1, 41, 49, 41, 1)
    if data == 'PNC':
        GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
        GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
        GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
        GlobalOpts.labelBaseString = get('DATA.LABELS')
        if GlobalOpts.pncDataType == 'AVG':
            GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.AVG_POOL{}'.format(GlobalOpts.dataScale))
        elif GlobalOpts.pncDataType == 'MAX':
            GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.MAX_PATH')
        elif GlobalOpts.pncDataType == 'NAIVE':
            GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.NAIVE{}'.format(GlobalOpts.dataScale))
        elif GlobalOpts.pncDataType == 'POOL_MIX':
            GlobalOpts.trainFiles = np.load(get('DATA.AUGMENTED.POOL_MIX_TRAIN_LIST_{}'.format(GlobalOpts.augRatio))).tolist()
            GlobalOpts.imageBaseString = get('DATA.AUGMENTED.POOL_MIX_PATH') + str(GlobalOpts.maxRatio) + "/"
            GlobalOpts.labelBaseString = get('DATA.AUGMENTED.POOL_MIX_LABELS')
        elif GlobalOpts.pncDataType == 'COMBINE':
            if GlobalOpts.origSize == 100:
                if GlobalOpts.augRatio >= 1:
                    GlobalOpts.trainFiles = np.load(get('DATA.AUGMENTED.COMBINE_TRAIN_LIST_{}_100'.format(int(GlobalOpts.augRatio)))).tolist()
                else:
                    GlobalOpts.trainFiles = np.load(get('DATA.AUGMENTED.COMBINE_TRAIN_LIST_0_{}_100'.format(int(100*GlobalOpts.augRatio)))).tolist()
            elif GlobalOpts.origSize == 200:
                if GlobalOpts.augRatio >= 1:
                    GlobalOpts.trainFiles = np.load(get('DATA.AUGMENTED.COMBINE_TRAIN_LIST_{}_200'.format(int(GlobalOpts.augRatio)))).tolist()
                else:
                    GlobalOpts.trainFiles = np.load(get('DATA.AUGMENTED.COMBINE_TRAIN_LIST_0_{}_200'.format(int(100*GlobalOpts.augRatio)))).tolist()
            elif GlobalOpts.origSize == 300:
                if GlobalOpts.augRatio >= 1:
                    GlobalOpts.trainFiles = np.load(get('DATA.AUGMENTED.COMBINE_TRAIN_LIST_{}_300'.format(int(GlobalOpts.augRatio)))).tolist()
                else:
                    GlobalOpts.trainFiles = np.load(get('DATA.AUGMENTED.COMBINE_TRAIN_LIST_0_{}_300'.format(int(100*GlobalOpts.augRatio)))).tolist()
            else:               
                if GlobalOpts.augRatio >= 1:
                    GlobalOpts.trainFiles = np.load(get('DATA.AUGMENTED.COMBINE_TRAIN_LIST_{}'.format(int(GlobalOpts.augRatio)))).tolist()
                else:
                    GlobalOpts.trainFiles = np.load(get('DATA.AUGMENTED.COMBINE_TRAIN_LIST_0_{}'.format(int(100*GlobalOpts.augRatio)))).tolist()
            GlobalOpts.imageBaseString = get('DATA.AUGMENTED.COMBINE_PATH')
            GlobalOpts.labelBaseString = get('DATA.AUGMENTED.COMBINE_LABELS')
        elif GlobalOpts.pncDataType == 'CONCAT':
            GlobalOpts.trainFiles = np.load(get('DATA.AUGMENTED.CONCAT_TRAIN_LIST')).tolist()
            GlobalOpts.imageBaseString = get('DATA.AUGMENTED.CONCAT_PATH')
            GlobalOpts.labelBaseString = get('DATA.AUGMENTED.CONCAT_LABELS')
            if GlobalOpts.testType == 'MAX':
                GlobalOpts.valdFiles = np.load(get('DATA.AUGMENTED.CONCAT_VALD_MAX')).tolist()
                GlobalOpts.testFiles = np.load(get('DATA.AUGMENTED.CONCAT_TEST_MAX')).tolist()
    elif data == 'PNC_GENDER':
        GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
        GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
        GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()
        GlobalOpts.imageBaseString = get('DATA.STRUCTURAL.AVG_POOL{}'.format(GlobalOpts.dataScale))
        GlobalOpts.labelBaseString = get('DATA.PHENOTYPICS.GENDER')
    elif data == 'ABIDE1':
        GlobalOpts.trainFiles = np.load(get('ABIDE.ABIDE1.TRAIN_LIST')).tolist()
        GlobalOpts.valdFiles = np.load(get('ABIDE.ABIDE1.VALD_LIST')).tolist()
        GlobalOpts.testFiles = np.load(get('ABIDE.ABIDE1.TEST_LIST')).tolist()
        GlobalOpts.imageBaseString = get('ABIDE.ABIDE1.AVG_POOL{}'.format(GlobalOpts.dataScale))
        GlobalOpts.labelBaseString = get('ABIDE.ABIDE1.LABELS')
    elif data == 'ABIDE2' or data == 'ABIDE2_AGE':
        baseString = 'ABIDE.ABIDE2.'
        if GlobalOpts.pheno:
            if GlobalOpts.listType == 'strat':
                baseString = '{}IQ_LISTS.STRAT_LISTS.'.format(baseString)
            elif GlobalOpts.listType == 'site':
                baseString = '{}IQ_LISTS.SITE_LISTS.'.format(baseString)
        GlobalOpts.trainFiles = np.load(get('{}TRAIN'.format(baseString))).tolist()
        GlobalOpts.valdFiles = np.load(get('{}VALD'.format(baseString))).tolist()
        GlobalOpts.testFiles = np.load(get('{}TEST'.format(baseString))).tolist()
        GlobalOpts.imageBaseString = get('ABIDE.ABIDE2.AVG_POOL{}'.format(GlobalOpts.dataScale))
        if 'AGE' in data:
            GlobalOpts.labelBaseString = get('ABIDE.ABIDE2.AGES')
        else:
            GlobalOpts.labelBaseString = get('ABIDE.ABIDE2.LABELS')
    elif 'ADHD' in data:
        if GlobalOpts.listType == 'strat':
            baseString = 'ADHD.STRAT_LISTS.'
        elif GlobalOpts.listType == 'site':
            baseString = 'ADHD.SITE_LISTS.'
        GlobalOpts.trainFiles = np.load(get('{}TRAIN'.format(baseString))).tolist()
        GlobalOpts.valdFiles = np.load(get('{}VALD'.format(baseString))).tolist()
        GlobalOpts.testFiles = np.load(get('{}TEST'.format(baseString))).tolist()
        GlobalOpts.imageBaseString = get('ADHD.AVG_POOL{}'.format(GlobalOpts.dataScale))
        if data == 'ADHD_AGE':
            GlobalOpts.labelBaseString = get('ADHD.AGES')
        else:
            GlobalOpts.labelBaseString = get('ADHD.LABELS')
    elif 'PAC' in data:
        baseString = 'PAC.'
        GlobalOpts.trainFiles = np.load(get('{}TRAIN'.format(baseString))).tolist()
        GlobalOpts.valdFiles = np.load(get('{}VALD'.format(baseString))).tolist()
        GlobalOpts.testFiles = np.load(get('{}TEST'.format(baseString))).tolist()
        GlobalOpts.imageBaseString = get('PAC.AVG_POOL{}'.format(GlobalOpts.dataScale))
        if 'AGE' in data:
            GlobalOpts.labelBaseString = get('PAC.AGES')
        else:
            GlobalOpts.labelBaseString = get('PAC.LABELS')
    GlobalOpts.numberTestItems = []
    GlobalOpts.numberValdItems = []
    for i in range(5):
        GlobalOpts.trainFiles[i] = np.array(GlobalOpts.trainFiles[i]).tolist()
        GlobalOpts.testFiles[i] = np.array(GlobalOpts.testFiles[i]).tolist()
        GlobalOpts.valdFiles[i] = np.array(GlobalOpts.valdFiles[i]).tolist()
        GlobalOpts.numberTestItems.append(len(GlobalOpts.testFiles))
        GlobalOpts.numberValdItems.append(len(GlobalOpts.valdFiles))
    GlobalOpts.name = '{}Scale{}Data{}Batch{}Rate{}'.format(GlobalOpts.type, GlobalOpts.scale, data, GlobalOpts.batchSize, GlobalOpts.learningRate)
    if GlobalOpts.sliceIndex is not None:
        GlobalOpts.name = '{}Slice{}'.format(GlobalOpts.name, GlobalOpts.sliceIndex)
    if GlobalOpts.align:
        GlobalOpts.name = '{}Aligned'.format(GlobalOpts.name)
    if GlobalOpts.padding is not None:
        GlobalOpts.name = '{}Padding{}'.format(GlobalOpts.name, GlobalOpts.padding)
    if GlobalOpts.regStrength is not None:
        GlobalOpts.name = '{}L2Reg{}'.format(GlobalOpts.name, GlobalOpts.regStrength)
    if GlobalOpts.maxNorm is not None:
        GlobalOpts.name = '{}MaxNorm{}'.format(GlobalOpts.name, GlobalOpts.maxNorm)
    if GlobalOpts.dropout is not None:
        GlobalOpts.name = '{}Dropout{}'.format(GlobalOpts.name, GlobalOpts.dropout)
    if GlobalOpts.skipConnection is not None:
        GlobalOpts.name = '{}SkipConnection{}'.format(GlobalOpts.name, GlobalOpts.skipConnection)
    if GlobalOpts.pncDataType == "POOL_MIX":
        GlobalOpts.name = '{}MAX_RATIO{}AUG_RATIO{}'.format(GlobalOpts.name, GlobalOpts.maxRatio, GlobalOpts.augRatio)
    if GlobalOpts.pncDataType == "COMBINE":
        GlobalOpts.name = '{}COMBINE_AUG_RATIO{}'.format(GlobalOpts.name, GlobalOpts.augRatio)
    if GlobalOpts.pncDataType == "CONCAT":
        if GlobalOpts.testType is None:
            GlobalOpts.testType = 'AVG'
        GlobalOpts.name = '{}CONCAT_TEST_WITH_{}'.format(GlobalOpts.name, GlobalOpts.testType)
    if GlobalOpts.augment is None:
        GlobalOpts.augment = 'none'
    else:
        if GlobalOpts.augRatio is None:
            GlobalOpts.augRatio == 1
        GlobalOpts.name = '{}AUGMENTED_BY_{}_AUG_RATIO{}'.format(GlobalOpts.name, GlobalOpts.augment, GlobalOpts.augRatio)
    GlobalOpts.summaryDir = '../summaries/{}/{}/'.format(summaryName,
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '../checkpoints/{}/{}/'.format(summaryName,
                                                     GlobalOpts.name)


def GetOps(labelsPL, outputLayer, learningRate=0.0001):
    """
    Given the Global Opts defined, returns a loss operation, an update operation,
    and summary operations.
    """
    if GlobalOpts.data == 'PNC' or 'AGE' in GlobalOpts.data:
        with tf.variable_scope('LossOperations'):
            lossOp = tf.losses.mean_squared_error(labels=labelsPL, predictions=outputLayer)
            MSEOp, MSEUpdateOp = tf.metrics.mean_squared_error(labels=labelsPL, predictions=outputLayer)
            MAEOp, MAEUpdateOp = tf.metrics.mean_absolute_error(labels=labelsPL, predictions=outputLayer)
            updateOp, gradients = GetTrainingOperation(lossOp, learningRate)
        printOps = PrintOps(ops=[MSEOp, MAEOp],
            updateOps=[MSEUpdateOp, MAEUpdateOp],
            names=['loss', 'MAE'],
            gradients=gradients)
    else:
        with tf.variable_scope('LossOperations'):
            if 'ADHD' in GlobalOpts.data:
                labelsPL = labelsPL >= 1
            oneHotLabels = tf.squeeze(tf.one_hot(indices=tf.cast(labelsPL, tf.int32), depth=2), axis=1)
            lossOp = tf.losses.softmax_cross_entropy(onehot_labels=oneHotLabels, logits=outputLayer)
            entropyOp, entropyUpdateOp = tf.metrics.mean(values=lossOp)
            labelClasses = tf.argmax(input=oneHotLabels, axis=1)
            predictionClasses = tf.argmax(input=outputLayer, axis=1)
            accuracyOp, accuracyUpdateOp, = tf.metrics.accuracy(labels=labelClasses, predictions=predictionClasses)
            aucOp, aucUpdateOp = tf.metrics.auc(labels=labelClasses, predictions=predictionClasses)
            errorOp = 1.0 - aucOp
            updateOp, gradients = GetTrainingOperation(lossOp, learningRate)
        printOps = PrintOps(ops=[errorOp, entropyOp, accuracyOp],
            updateOps=[aucUpdateOp, entropyUpdateOp, accuracyUpdateOp],
            names=['loss', 'entropy', 'Accuracy'],
            gradients=gradients)
    return lossOp, printOps, updateOp

def GetArgs():
    additionalArgs = [
        {
        'flag': '--scale',
        'help': 'The scale at which to slice dimensions. For example, a scale of 2 means that each dimension will be devided into 2 distinct regions, for a total of 8 contiguous chunks.',
        'action': 'store',
        'type': int,
        'dest': 'scale',
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
        'flag': '--summaryName',
        'help': 'The file name to put the results of this run into.',
        'action': 'store',
        'type': str,
        'dest': 'summaryName',
        'required': True
        },
        {
        'flag': '--data',
        'help': 'One of: PNC, PNC_GENDER, ABIDE1, ABIDE2, ABIDE2_AGE, PAC, PAC_AGE',
        'action': 'store',
        'type': str,
        'dest': 'data',
        'required': True
        },
        {
        'flag': '--poolType',
        'help': 'One of MAX, AVG, STRIDED, NONE. Type of pooling layer used inside the network. Default is max pooling.',
        'action': 'store',
        'type': str,
        'dest': 'poolType',
        'required': False,
        'const': None
        },
        {
        'flag': '--sliceIndex',
        'help': 'Set this to an integer to select a single brain region as opposed to concatenating all regions along the depth channel.',
        'action': 'store',
        'type': int,
        'dest': 'sliceIndex',
        'required': False,
        'const': None
        },
        {
        'flag': '--align',
        'help': 'Set to true to align channels.',
        'action': 'store',
        'type': int,
        'dest': 'align',
        'required': False,
        'const': None
        },
        {
        'flag': '--numberTrials',
        'help': 'Number of repeated models to run.',
        'action': 'store',
        'type': int,
        'dest': 'numberTrials',
        'required': False,
        'const': None
        },
        {
        'flag': '--padding',
        'help': 'Set this to an integer to crop the image to the brain and then apply `padding` amount of padding.',
        'action': 'store',
        'type': int,
        'dest': 'padding',
        'required': False,
        'const': None
        },
        {
        'flag': '--batchSize',
        'help': 'Batch size to train with. Default is 4.',
        'action': 'store',
        'type': int,
        'dest': 'batchSize',
        'required': False,
        'const': None
        },
        {
        'flag': '--pheno',
        'help': 'Specify 1 to add phenotypics to the model.',
        'action': 'store',
        'type': int,
        'dest': 'pheno',
        'required': False,
        'const': None
        },
        {
        'flag': '--validationDir',
        'help': 'Checkpoint directory to restore the model from.',
        'action': 'store',
        'type': str,
        'dest': 'validationDir',
        'required': False,
        'const': None
        },
        {
        'flag': '--regStrength',
        'help': 'Lambda value for L2 regularization. If not specified, no regularization is applied.',
        'action': 'store',
        'type': float,
        'dest': 'regStrength',
        'required': False,
        'const': None
        },
        {
        'flag': '--learningRate',
        'help': 'Global optimization learning rate. Default is 0.0001.',
        'action': 'store',
        'type': float,
        'dest': 'learningRate',
        'required': False,
        'const': None
        },
        {
        'flag': '--maxNorm',
        'help': 'Specify an integer to constrain kernels with a maximum norm.',
        'action': 'store',
        'type': int,
        'dest': 'maxNorm',
        'required': False,
        'const': None
        },
        {
        'flag': '--dropout',
        'help': 'The probability of keeping a neuron alive during training. Defaults to 0.6.',
        'action': 'store',
        'type': float,
        'dest': 'dropout',
        'required': False,
        'const': None
        },
        {
        'flag': '--dataScale',
        'help': 'The downsampling rate of the data. Either 1, 2 or 3. Defaults to 3. ',
        'action': 'store',
        'type': int,
        'dest': 'dataScale',
        'required': False,
        'const': None
        },
        {
        'flag': '--pncDataType',
        'help': 'One of AVG, MAX, NAIVE, POOL_MIX, COMBINE. Defaults to AVG. If set, dataScale cannot be specified.',
        'action': 'store',
        'type': str,
        'dest': 'pncDataType',
        'required': False,
        'const': None
        },
        {
        'flag': '--listType',
        'help': 'Only valid for ABIDE and ADHD. One of strat or site.',
        'action': 'store',
        'type': str,
        'dest': 'listType',
        'required': False,
        'const': None
        },
        {
        'flag': '--depthwise',
        'help': 'Set to 1 to use depthwise convolutions for the entire network.',
        'action': 'store',
        'type': int,
        'dest': 'depthwise',
        'required': False,
        'const': None
        },
        {
        'flag': '--hiddenUnits',
        'help': 'The number of hidden units. Defaults to 256.',
        'action': 'store',
        'type': int,
        'dest': 'hiddenUnits',
        'required': False,
        'const': None
        },
        {
        'flag': '--trainingSize',
        'help': 'Number of training examples. Default is 524, which is max.',
        'action': 'store',
        'type': int,
        'dest': 'trainingSize',
        'required': False,
        'const': None
        },
        {
        'flag': '--skipConnection',
        'help': 'Set to 1 to allow skip connection layer, add residuals to the network (like ResNet).',
        'action': 'store',
        'type': int,
        'dest': 'skipConnection',
        'required': False,
        'const': None
        },
        {
        'flag': '--maxRatio',
        'help': 'Ratio of max pooling in the pool_mix augmentation. Default to 0.25.',
        'action': 'store',
        'type': float,
        'dest': 'maxRatio',
        'required': False,
        'const': None
        },
        {
        'flag': '--augRatio',
        'help': 'Ratio of augmented images versus pure average images in the pool_mix and combine augmentation. Default to 2.',
        'action': 'store',
        'type': float,
        'dest': 'augRatio',
        'required': False,
        'const': None
        },
        {
        'flag': '--testType',
        'help': 'One of AVG, MAX. Type of validation and test file preprocessing setting. Default to AVG.',
        'action': 'store',
        'type': str,
        'dest': 'testType',
        'required': False,
        'const': None
        },
        {
        'flag': '--augment',
        'help': 'One of FLIP, TRANSLATE. Type of standard augmentation. Default to None.',
        'action': 'store',
        'type': str,
        'dest': 'augment',
        'required': False,
        'const': None
        },
        {
        'flag': '--origSize',
        'help': 'Size of the original sample before augmentation. One of 100, 200, 300. If None, then all samples are used.',
        'action': 'store',
        'type': int,
        'dest': 'origSize',
        'required': False,
        'const': None
        }
        ]
    ParseArgs('Run 3D CNN over structural MRI volumes', additionalArgs=additionalArgs)
    if GlobalOpts.numberTrials is None:
        GlobalOpts.numberTrials = 5
    if GlobalOpts.batchSize is None:
        GlobalOpts.batchSize = 4
    if GlobalOpts.learningRate is None:
        GlobalOpts.learningRate = 0.0001
    if GlobalOpts.dropout is None:
        GlobalOpts.dropout = 0.6
    if GlobalOpts.dataScale is None:
        GlobalOpts.dataScale = 3
    if GlobalOpts.pncDataType is None:
        GlobalOpts.pncDataType = 'AVG'
    if GlobalOpts.listType is None:
        GlobalOpts.listType = 'strat'
    if GlobalOpts.hiddenUnits is None:
        GlobalOpts.hiddenUnits = 256
    if GlobalOpts.poolType is None:
        GlobalOpts.poolType = 'MAX'
    if GlobalOpts.pncDataType == 'POOL_MIX' and GlobalOpts.maxRatio is None:
        GlobalOpts.maxRatio = 0.25
    if GlobalOpts.pncDataType == 'POOL_MIX' and GlobalOpts.augRatio is None:
        GlobalOpts.augRatio = 2

def compareCustomCNN(validate=False):
    GetArgs()
    DefineDataOpts(data=GlobalOpts.data, summaryName=GlobalOpts.summaryName)
    if GlobalOpts.trainingSize is not None:
        GlobalOpts.trainFiles = GlobalOpts.trainFiles[:GlobalOpts.trainingSize]
    modelTrainer = ModelTrainer()
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    imagesPL, labelsPL = StructuralPlaceholders(GlobalOpts.imageBatchDims)
    trainingPL = TrainingPlaceholder()

    if GlobalOpts.type == 'traditional':
        convLayers = [8, 16, 32, 64]
    elif GlobalOpts.type == 'reverse':
        convLayers = [64, 32, 16, 8]
    if GlobalOpts.data == 'PNC' or 'AGE' in GlobalOpts.data:
        fullyConnectedLayers = [GlobalOpts.hiddenUnits, 1]
    else:
        fullyConnectedLayers = [GlobalOpts.hiddenUnits, 2]
    if GlobalOpts.pheno:
        # FIXME: new dataset cannot accomodate phenotypic
        phenotypicBaseStrings=[
            '/data1/brain/ABIDE/ABIDE2/gender/',
            '/data1/brain/ABIDE/ABIDE2/IQData/FIQ/',
            '/data1/brain/ABIDE/ABIDE2/IQData/VIQ/',
            '/data1/brain/ABIDE/ABIDE2/IQData/PIQ/'
        ]
        if GlobalOpts.data != 'ABIDE2_AGE':
            phenotypicBaseStrings.append('/data1/brain//ABIDE/ABIDE2/ages/')
        phenotypicsPL = tf.placeholder(dtype=tf.float32, shape=(None, len(phenotypicBaseStrings) + 1), name='phenotypicsPL')
        trainDataSet.CreatePhenotypicOperations(phenotypicBaseStrings)
        valdDataSet.CreatePhenotypicOperations(phenotypicBaseStrings)
        testDataSet.CreatePhenotypicOperations(phenotypicBaseStrings)
    else:
        phenotypicsPL = None

    for vald in valdDataSet:
        vald.PreloadData()

    """
    Depthwise is used for depthwise convolution network.
    Implemented by Pascal Sturmfels
    """
    if GlobalOpts.depthwise:
        if GlobalOpts.type == 'traditional':
            convLayers = [1, 2, 4, 8]
        elif GlobalOpts.type == 'reverse':
            convLayers = [8, 4, 2, 1]
        outputLayer = separableCNN(imagesPL,
                                  trainingPL,
                                  GlobalOpts.scale,
                                  convLayers,
                                  fullyConnectedLayers,
                                  keepProbability=GlobalOpts.dropout)
    else:
        outputLayer = customCNN(imagesPL,
                                trainingPL,
                                GlobalOpts.scale,
                                convLayers,
                                fullyConnectedLayers,
                                keepProbability=GlobalOpts.dropout,
                                poolType=GlobalOpts.poolType,
                                sliceIndex=GlobalOpts.sliceIndex,
                                align=GlobalOpts.align,
                                padding=GlobalOpts.padding,
                                phenotypicsPL=phenotypicsPL,
                                skipConnection=GlobalOpts.skipConnection)

    lossOp, printOps, updateOp = GetOps(labelsPL, outputLayer, learningRate=GlobalOpts.learningRate)
    modelTrainer.DefineNewParams(GlobalOpts.summaryDir,
                                GlobalOpts.checkpointDir,
                                imagesPL,
                                trainingPL,
                                labelsPL,
                                trainDataSet,
                                valdDataSet,
                                testDataSet,
                                GlobalOpts.numSteps,
                                phenotypicsPL=phenotypicsPL)
    config  = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GlobalOpts.gpuMemory
    with tf.Session(config=config) as sess:
        if validate:
            modelTrainer.getPatientPerformances(sess,
                                                outputLayer,
                                                name=GlobalOpts.name,
                                                numIters=GlobalOpts.numberTrials)
            #modelTrainer.ValidateModel(sess,
            #                      updateOp,
            #                      printOps,
            #                      name=GlobalOpts.name,
            #                      numIters=GlobalOpts.numberTrials)
        else:
            modelTrainer.RepeatTrials(sess,
                                  updateOp,
                                  printOps,
                                  name=GlobalOpts.name,
                                  numIters=GlobalOpts.numberTrials)

