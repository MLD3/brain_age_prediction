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

def get_weights(saver, sess):
    if is_file_prefix('TRAIN.CNN.CHECKPOINT'):
        saver.restore(sess, get('TRAIN.CNN.CHECKPOINT'))
        print('Restored weights from a checkpoint.')
    else:
        print('Training from scratch...')

def save_model(sess, path):
    saver = tf.train.Saver()
    save_path = saver.save(sess, path)
    print("Model saved to file: " + str(save_path))

def report_training_progress(sess, batch_index, input_layer, loss_func, validationSet):
    if batch_index % 5:
        return
    print('starting batch number %d \033[100D\033[1A' % batch_index)
    if batch_index % 50:
        return
    error = loss_func.eval(feed_dict={input_layer: validationSet.images, true_ages: validationSet.labels})
    print('\n \t Evaluated Loss Function: %f' % error)
    if batch_index % 500:
        return
    print("Saving model...")
    save_model(sess, get('TRAIN.CNN.CHECKPOINT'))


def trainCNN(sess, input_layer, prediction_layer, loss_func, optimizer, trainingSet, validationSet):
    try:
        for batch_index in range(get('TRAIN.CNN.NB_STEPS')):
            report_training_progress(sess,
                batch_index, input_layer, loss_func, validationSet)
            batch_images, batch_labels = trainingSet.next_batch(
                get('TRAIN.CNN.BATCH_SIZE'))
            optimizer.run(
                feed_dict={input_layer: batch_images, true_ages: batch_labels})
    except KeyboardInterrupt:
        print('Terminating training session due to keyboard exception...')

#TODO: Default k=5
def crossValidate(sess, trainingSet, input_layer, prediction_layer, loss_func, optimizer, k=2, numReps=1):
    validationPerformance = []
    skf = KFold(n_splits=k)
    for trial in range(numReps):
        for trainIndex, valdIndex in skf.split(trainingSet.images, trainingSet.labels):
            X_train, X_vald = trainingSet.images[trainIndex], trainingSet.images[valdIndex]
            y_train, y_vald = trainingSet.labels[trainIndex], trainingSet.labels[valdIndex]
            splitTrainSet = DataSet(X_train, y_train)
            validationSet = DataSet(X_vald, y_vald)
            trainCNN(sess, input_layer, prediction_layer, loss_func, optimizer, splitTrainSet, validationSet)
            perf = loss_func.eval(feed_dict={input_layer: X_vald, true_ages: y_vald})
            validationPerformance.append(perf)
    return np.mean(validationPerformance)

#TODO: default numTestSplits=2
def performanceOnParameter(trainSet, sess, input_layer, prediction_layer, loss_func, optimizer):
    numTestSplits = 1
    for trial in range(numTestSplits):
        X_train, X_test, y_train, y_test = train_test_split(trainSet.images, trainSet.labels, test_size=0.2)
        splitTrainSet = DataSet(X_train, y_train)
        splitTestSet = DataSet(X_test, y_test)

        valPerf = crossValidate(sess, splitTrainSet, input_layer, prediction_layer, loss_func, optimizer)
        trainCNN(sess, input_layer, prediction_layer, loss_func, optimizer, splitTrainSet, splitTestSet)
        error = loss_func.eval(feed_dict={input_layer: splitTestSet.images, true_ages: splitTestSet.labels})
        print("TEST performance with given loss was: " + '%f' % error)

def test1(trainSet):
    print("=========================================")
    print("First architecture: 3x3 filters, relu, 3x3 pooling, stride=1, small init biases and sd")
    print("=========================================")
    input_layer, prediction_layer = cnnDefault()
    true_ages = tf.placeholder(tf.float32, shape=[None, 1])
    rmse = tf.sqrt(tf.losses.mean_squared_error(labels=true_ages, predictions=prediction_layer))
    optimizer = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(rmse)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # get_weights(saver, sess)

    performanceOnParameter(trainSet, sess, input_layer, prediction_layer, rmse, optimizer)

def test2(trainSet):
    print("=========================================")
    print("Second architecture: 3x3 filters, relu, 3x3 pooling, stride=1, large init biases and sd")
    print("=========================================")
    input_layer, prediction_layer = cnnDefault(meanDefault=0.0, sdDefault=0.1, biasDefault=1.0)
    true_ages = tf.placeholder(tf.float32, shape=[None, 1])
    rmse = tf.sqrt(tf.losses.mean_squared_error(labels=true_ages, predictions=prediction_layer))
    optimizer = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(rmse)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # get_weights(saver, sess)

    performanceOnParameter(trainSet, sess, input_layer, prediction_layer, rmse, optimizer)

def test3(trainSet):
    print("=========================================")
    print("Third architecture: 3x3 filters, relu, 3x3 pooling, stride=1, positive initial means")
    print("=========================================")
    input_layer, prediction_layer = cnnDefault(meanDefault=0.1, sdDefault=0.1, biasDefault=1.0)
    true_ages = tf.placeholder(tf.float32, shape=[None, 1])
    rmse = tf.sqrt(tf.losses.mean_squared_error(labels=true_ages, predictions=prediction_layer))
    optimizer = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(rmse)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # get_weights(saver, sess)

    performanceOnParameter(trainSet, sess, input_layer, prediction_layer, rmse, optimizer)

def test4(trainSet):
    print("=========================================")
    print("Fourth architecture: 4x4 filters and varying stride, relu, 3x3 pooling")
    print("=========================================")
    input_layer, prediction_layer = cnnSmall()
    true_ages = tf.placeholder(tf.float32, shape=[None, 1])
    rmse = tf.sqrt(tf.losses.mean_squared_error(labels=true_ages, predictions=prediction_layer))
    optimizer = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(rmse)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # get_weights(saver, sess)

    performanceOnParameter(trainSet, sess, input_layer, prediction_layer, rmse, optimizer)

def test5(trainSet):
    print("=========================================")
    print("Fifth architecture: large learning rate")
    print("=========================================")
    input_layer, prediction_layer = cnnDefault()
    true_ages = tf.placeholder(tf.float32, shape=[None, 1])
    rmse = tf.sqrt(tf.losses.mean_squared_error(labels=true_ages, predictions=prediction_layer))
    optimizer = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE_LARGE')).minimize(rmse)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # get_weights(saver, sess)

    performanceOnParameter(trainSet, sess, input_layer, prediction_layer, rmse, optimizer)

def test6(trainSet):
    print("=========================================")
    print("Sixth architecture: SGD + Momentum optimizer, momentum=0.9")
    print("=========================================")
    input_layer, prediction_layer = cnnDefault()
    true_ages = tf.placeholder(tf.float32, shape=[None, 1])
    rmse = tf.sqrt(tf.losses.mean_squared_error(labels=true_ages, predictions=prediction_layer))
    optimizer = tf.train.MomentumOptimizer(get('TRAIN.CNN.LEARNING_RATE'), get('TRAIN.CNN.MOMENTUM')).minimize(rmse)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # get_weights(saver, sess)

    performanceOnParameter(trainSet, sess, input_layer, prediction_layer, rmse, optimizer)

if __name__ == '__main__':
    ##########################################################################
    ############################# GET DATA SETS ##############################
    ##########################################################################
    dataHolder = DataHolder(readCSVData(get('DATA.PHENOTYPICS.PATH')))
    dataHolder.getMatricesFromPath(get('DATA.MATRICES.PATH'))
    dataHolder.matricesToImages()
    trainSet = dataHolder.returnDataSet()

    # print("=========================================")
    # print("First architecture: 3x3 filters, relu, 3x3 pooling, stride=1, small init biases and sd")
    # print("=========================================")
    # input_layer, prediction_layer = cnnDefault()
    # true_ages = tf.placeholder(tf.float32, shape=[None, 1])
    # rmse = tf.sqrt(tf.losses.mean_squared_error(labels=true_ages, predictions=prediction_layer))
    # optimizer = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(rmse)
    #
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # # saver = tf.train.Saver()
    # # get_weights(saver, sess)
    #
    # performanceOnParameter(trainSet, sess, input_layer, prediction_layer, rmse, optimizer)
    #
    #
    # print("=========================================")
    # print("Second architecture: 3x3 filters, relu, 3x3 pooling, stride=1, large init biases and sd")
    # print("=========================================")
    # input_layer, prediction_layer = cnnDefault(meanDefault=0.0, sdDefault=0.1, biasDefault=1.0)
    # true_ages = tf.placeholder(tf.float32, shape=[None, 1])
    # rmse = tf.sqrt(tf.losses.mean_squared_error(labels=true_ages, predictions=prediction_layer))
    # optimizer = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(rmse)
    #
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # # saver = tf.train.Saver()
    # # get_weights(saver, sess)
    #
    # performanceOnParameter(trainSet, sess, input_layer, prediction_layer, rmse, optimizer)
    #
    # print("=========================================")
    # print("Third architecture: 3x3 filters, relu, 3x3 pooling, stride=1, positive initial means")
    # print("=========================================")
    # input_layer, prediction_layer = cnnDefault(meanDefault=0.1, sdDefault=0.1, biasDefault=1.0)
    # true_ages = tf.placeholder(tf.float32, shape=[None, 1])
    # rmse = tf.sqrt(tf.losses.mean_squared_error(labels=true_ages, predictions=prediction_layer))
    # optimizer = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(rmse)
    #
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # # saver = tf.train.Saver()
    # # get_weights(saver, sess)
    #
    # performanceOnParameter(trainSet, sess, input_layer, prediction_layer, rmse, optimizer)
    #
    # print("=========================================")
    # print("Fourth architecture: 4x4 filters and varying stride, relu, 3x3 pooling")
    # print("=========================================")
    # input_layer, prediction_layer = cnnSmall()
    # true_ages = tf.placeholder(tf.float32, shape=[None, 1])
    # rmse = tf.sqrt(tf.losses.mean_squared_error(labels=true_ages, predictions=prediction_layer))
    # optimizer = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE')).minimize(rmse)
    #
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # # saver = tf.train.Saver()
    # # get_weights(saver, sess)
    #
    # performanceOnParameter(trainSet, sess, input_layer, prediction_layer, rmse, optimizer)
    #
    # print("=========================================")
    # print("Fifth architecture: large learning rate")
    # print("=========================================")
    # input_layer, prediction_layer = cnnDefault()
    # true_ages = tf.placeholder(tf.float32, shape=[None, 1])
    # rmse = tf.sqrt(tf.losses.mean_squared_error(labels=true_ages, predictions=prediction_layer))
    # optimizer = tf.train.AdamOptimizer(get('TRAIN.CNN.LEARNING_RATE_LARGE')).minimize(rmse)
    #
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # # saver = tf.train.Saver()
    # # get_weights(saver, sess)
    #
    # performanceOnParameter(trainSet, sess, input_layer, prediction_layer, rmse, optimizer)


    print("=========================================")
    print("Sixth architecture: SGD + Momentum optimizer, momentum=0.9")
    print("=========================================")
    input_layer, prediction_layer = cnnDefault()
    true_ages = tf.placeholder(tf.float32, shape=[None, 1])
    rmse = tf.sqrt(tf.losses.mean_squared_error(labels=true_ages, predictions=prediction_layer))
    optimizer = tf.train.MomentumOptimizer(get('TRAIN.CNN.LEARNING_RATE'), get('TRAIN.CNN.MOMENTUM')).minimize(rmse)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # get_weights(saver, sess)

    performanceOnParameter(trainSet, sess, input_layer, prediction_layer, rmse, optimizer)
