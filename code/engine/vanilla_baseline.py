import numpy as np
import pandas as pd
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from utils.config import get, is_file_prefix
from model.fMRI_cnn import *
from data_scripts.DataSet import DataSet


def DefineFeedDict(dataSet, imagesPL, labelsPL):
    feed_dict = {
        imagesPL: dataSet.images,
        labelsPL: dataSet.labels
    }
    return feed_dict

def GetEvaluatedLoss(sess, dataSet, lossFunction, imagesPL, labelsPL):
    feed_dict = DefineFeedDict(dataSet, imagesPL, labelsPL)
    return sess.run(lossFunction, feed_dict=feed_dict)

# def ReportProgress(sess, step, lossFunction, imagesPL, labelsPL, splitTrainSet, splitTestSet):
def ReportProgress(sess, step, lossFunction, trainOperation):    
    if step % 10 == 0:
        trainFeedDict = DefineFeedDict(splitTrainSet, imagesPL, labelsPL)
        trainingLoss = GetEvaluatedLoss(sess, splitTrainSet, lossFunction, imagesPL, labelsPL)
        # _, trainingLoss = sess.run([trainOperation, lossFunction])
        # print("Training Loss: ", trainingLoss)
        testFeedDict = DefineFeedDict(splitTestSet, imagesPL, labelsPL)
        validationLoss = GetEvaluatedLoss(sess, splitTestSet, lossFunction, imagesPL, labelsPL)
        print('Step: %d, Evaluated Training Loss: %f, Evaluated Test Loss: %f' % (step, trainingLoss, validationLoss))


# def TrainModel(sess, dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction):
def TrainModel(train_dataSet, test_dataSet, numRepeats = 1):    
    X_train, _, y_train, _ = train_test_split(train_dataSet.images, train_dataSet.labels, test_size=0.5)
    # X_train = train_dataSet.images
    # y_train = train_dataSet.labels

    X_test = test_dataSet.images
    y_test = test_dataSet.labels
    # splitTrainSet = DataSet(X_train, y_train)
    # splitTestSet = DataSet(X_test, y_test)

    # for batch_index in range(get('TRAIN.VANILLA_BASELINE.NB_STEPS')):
    #     # batch_images, batch_labels = tf.train.shuffle_batch([X_train, X_test], batch_size = get('TRAIN.VANILLA_BASELINE.BATCH_SIZE'), capacity = get('TRAIN.VANILLA_BASELINE.CAPACITY'), min_after_dequeue = get('TRAIN.VANILLA_BASELINE.MIN_AFTER_DEQUEUE'))
    #     batch_images, batch_labels = splitTrainSet.next_batch(
    #         get('TRAIN.VANILLA_BASELINE.BATCH_SIZE'))
    #     feed_dict = DefineFeedDict(DataSet(batch_images, batch_labels), imagesPL, labelsPL)
    #     # tf.train.start_queue_runners(sess = sess)
    #     sess.run(trainOperation, feed_dict=feed_dict)
    #     # sess.run([batch_images, batch_labels])
    #     # sess.run(trainOperation)
    #     ReportProgress(sess, batch_index, lossFunction, imagesPL, labelsPL, splitTrainSet, splitTestSet)
    #     # ReportProgress(sess, batch_index, lossFunction, trainOperation)

    # trainingLoss = GetEvaluatedLoss(sess, splitTrainSet, lossFunction, imagesPL, labelsPL)
    # testLoss = GetEvaluatedLoss(sess, splitTestSet, lossFunction, imagesPL, labelsPL)
    # return (trainingLoss, testLoss)
    # # return trainingLoss

    # queue to load data
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_train.shape[1:])
    # print(y_train.shape[1:])
    # exit(0)
    q = tf.FIFOQueue(capacity = get('TRAIN.VANILLA_BASELINE.CAPACITY'), dtypes = [tf.float32, tf.float32], shapes = [X_train.shape[1:], y_train.shape[1:]])
    enqueue_op = q.enqueue_many([X_train, y_train])
    number_of_threads = 1
    qr = tf.train.QueueRunner(q, [enqueue_op] * number_of_threads)
    tf.train.add_queue_runner(qr)
    batch_images, batch_labels = q.dequeue_many(get('TRAIN.VANILLA_BASELINE.BATCH_SIZE'))

    meanDefault = 0.0
    sdDefault = 0.01
    biasDefault = 0.1

    with tf.variable_scope('4D_CNN'):
        prediction = model_architecture(batch_images)

    with tf.variable_scope('Loss'):
        lossFunction = tf.losses.mean_squared_error(labels=batch_labels, predictions=prediction)
        print_loss = tf.Print(lossFunction, data = [lossFunction], message = "MSE loss: ")

    with tf.variable_scope('4D_CNN', reuse = True):
        test_prediction = model_architecture(X_test)

    # with tf.variable_scope('Test_loss'):
        test_loss = tf.losses.mean_squared_error(labels = y_test, predictions = test_prediction)
        # print_test_loss = tf.Print(test_loss, data = [test_loss], message = "Test MSE loss: ")


    global_step = tf.Variable(0, name='global_step', trainable=False)
    trainOperation = tf.train.AdamOptimizer(get('TRAIN.VANILLA_BASELINE.LEARNING_RATE')).minimize(lossFunction, global_step=global_step)

    mse = []
    test_mse = []
    for i in range(numRepeats):
        with tf.Session() as sess:
            print("Start tf session")
            init = tf.global_variables_initializer()
            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)

            sess.run(print_loss)
            # print("Before training, MSE is ", loss)
            print("Start looping")
            for step in range(get('TRAIN.VANILLA_BASELINE.NB_STEPS')):
                loss = sess.run(trainOperation)
                # print(sess.run(lossFunction))
           
                if step % 10 == 0:
                    print("At step " + str(step) + " MSE is " + str(sess.run(lossFunction)))
                    print("At step " + str(step) + " test MSE is " + str(sess.run(test_loss)))
                    # sess.run(print_loss)
                    # print("At step " + str(step) + " training MSE is " + str(loss))
            mse.append(sess.run(lossFunction))
            test_mse.append(sess.run(test_loss))
            coord.request_stop()
            coord.join(threads)
    print(mse)
    print(test_mse)

def model_architecture(batch_images):
    # 52*62*45 -> 52*62*45*2
    conv1 = conv_layer(x = batch_images, input_height = 52, input_width = 62, 
                        input_depth = 45, input_channels = 1, filter_size = 7, 
                        output_channels=2, padding='SAME', act_type='relu', 
                        mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=1)
    # 52*62*45*2 -> 26*31*23*2
    pool1 = pool_layer(x=conv1, input_height = 52, input_width = 62, input_depth = 45, input_channels=2, k_length=2, stride=2, padding='SAME')
    # 26*31*23*2 -> 26*31*23*8
    conv2 = conv_layer(x=pool1, input_height = 26, input_width = 31, input_depth = 23, input_channels=2,
                        filter_size=5, output_channels=8, padding='SAME',
                        act_type='relu', mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=1)
    # 26*31*23*8 -> 13*16*12*8
    pool2 = pool_layer(x=conv2, input_height = 26, input_width = 31, input_depth = 23, input_channels=8, k_length=2, stride=2, padding='SAME')
    # 13*16*12*8 -> 13*16*12*16
    conv3 = conv_layer(x=pool2, input_height = 13, input_width = 16, input_depth = 12, input_channels=8,
                        filter_size=3, output_channels=16, padding='SAME',
                        act_type='relu', mean=meanDefault, sd=sdDefault, bias=biasDefault, stride=1)
    # 13*16*12*16 -> 7*8*6*16
    pool3 = pool_layer(x=conv3, input_height = 13, input_width = 16, input_depth = 12, input_channels=16, k_length=3, stride=2, padding='SAME')
    # fully connected
    prediction = fully_connected_layer(pool3, shape=[7 * 8 * 6 * 16, 1], act_type='none', mean=meanDefault, sd=sdDefault, bias=biasDefault)
    return prediction


def RepeatModel(dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, numRepeats=10):
    trainingLosses = []
    testLosses = []
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    for d in ['/gpu:0','/gpu:1']:
        with tf.device(d):    
            for i in range(numRepeats):
                with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    (trainingLoss, testLoss) = TrainModel(sess, dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction)
                    trainingLosses.append(trainingLoss)
                    testLosses.append(testLoss)
    print("Mean Evaluated Training Loss: %f" % np.mean(trainingLosses))
    print("SD   Evaluated Training Loss: %f" % np.std(trainingLosses))
    print("Mean Evaluated Test Loss: %f" % np.mean(testLosses))
    print("SD   Evaluated Test Loss: %f" % np.std(testLosses))


def test(dataSet):
    print('----------------------------------------------------------------')
    print('----------------------------TEST 1------------------------------')
    print('----------------------------------------------------------------')
    # imagesPL, predictionLayer = fMRI_4D_CNN()
    # labelsPL = tf.placeholder(tf.float32, shape=[None, 1])
    # total_error = tf.reduce_sum(tf.square(tf.subtract(labelsPL, tf.reduce_mean(labelsPL))))
    # unexplained_error = tf.reduce_sum(tf.square(tf.subtract(labelsPL, predictionLayer)))
    # lossFunction = tf.div(total_error, unexplained_error)
    # lossFunction = tf.losses.mean_squared_error(labels=labelsPL, predictions=predictionLayer)
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    # trainOperation = tf.train.AdamOptimizer(get('TRAIN.VANILLA_BASELINE.LEARNING_RATE')).minimize(lossFunction, global_step=global_step)

    # RepeatModel(dataSet, imagesPL, labelsPL, predictionLayer, trainOperation, lossFunction, numRepeats=3)
    TrainModel(dataSet)

if __name__ == '__main__':
    train_dataHolder = DataHolder(readCSVData(get('DATA.SAMPLE.TRAIN_PATH')))
    train_dataHolder.getNIIImagesFromPath(get('DATA.IMAGES.TRAIN_PATH'))
    # print(len(dataHolder.matrices))
    train_dataSet = train_dataHolder.returnNIIDataset()

    test_dataHolder = DataHolder(readCSVData(get('DATA.SAMPLE.TEST_PATH')))
    test_dataHolder.getNIIImagesFromPath(get('DATA.IMAGES.TEST_PATH'))
    test_dataSet = test_dataHolder.returnNIIDataset()
    # print(dataSet.images.shape)
    # test(dataSet)

    TrainModel(train_dataSet, test_dataSet, numRepeats = 5)
