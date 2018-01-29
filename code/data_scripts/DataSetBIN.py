import tensorflow as tf
import numpy as np
from utils.config import get

class DataSetBIN(object):
    def __init__(self,
            binFileNames,
            imageDims=[145, 145, 1],
            bytesPerNumber=8,
            decodeType=tf.float64,
            numReaderThreads=16,
            batchSize=32,
            maxItemsInQueue=2000,
            minItemsInQueue=500,
            shuffle=True,
            training=True
        ):
        # Define a file name queue
        self.binFileNames = binFileNames
        fileNameQueue = tf.train.string_input_producer(binFileNames)

        # Define the record reader
        numbersInImage = np.prod(imageDims)
        totalLineBytes = (1 + numbersInImage) * bytesPerNumber
        reader = tf.FixedLengthRecordReader(record_bytes=totalLineBytes)
        key, value = reader.read(fileNameQueue)
        decoded = tf.decode_raw(value, decodeType)

        # Define the label operation. Note: making the assumption
        # that the label is represented by a single number, which is
        # true of most regression and classification tasks
        label = tf.strided_slice(decoded, [0], [1])
        image = tf.reshape(tf.strided_slice(decoded, [1], [1 + numbersInImage]),
                           imageDims)
        label.set_shape((1,))

        if not training:
            image = tf.pad(image, [[24, 0], [0, 0], [24, 0], [0, 0]], name='Image Padding', mode='CONSTANT')

        # Define the batch operations
        if shuffle:
            self.imageBatchOperation, self.labelBatchOperation = tf.train.shuffle_batch(
                [image, label],
                batch_size=batchSize,
                num_threads=numReaderThreads,
                capacity=maxItemsInQueue,
                min_after_dequeue=minItemsInQueue
            )
        else:
            self.imageBatchOperation, self.labelBatchOperation = tf.train.batch(
                [image, label],
                batch_size=batchSize,
                num_threads=numReaderThreads,
                capacity=maxItemsInQueue
            )

    def GetBatchOperations(self):
        return self.imageBatchOperation, self.labelBatchOperation

    def GetRandomResamples(self, batchSize=100):
        """
        Returns an operation that randomly samples the contained constant images/labels.
        """
        randomIndices = tf.random_uniform(shape=(batchSize,), minval=0, maxval=batchSize, dtype=tf.int32)
        randomImages = tf.gather(self.imageBatchOperation, randomIndices)
        randomLabels = tf.gather(self.labelBatchOperation, randomIndices)
        return randomImages, randomLabels

    ##### THE FUNCTIONS BELOW ARE DEPRECATED #####
    def NextBatch(self, sess):
        """
        Returns the next batch of examples using sess. Assumes that
        tf.train.start_queue_runners has already been called. If
        it hasn't been called, this code will not execute properly.
        """
        return sess.run([self.imageBatchOperation, self.labelBatchOperation])

    def GetConstantDataVariables(self, inType=tf.float64, dataShape=(5000, 145, 145, 1), labelShape=(5000, 1)):
        self.constantImageVar = tf.Variable(self.imageBatchOperation, trainable=False, collections=[], name='imageVariable')
        self.constantLabelVar = tf.Variable(self.labelBatchOperation, trainable=False, collections=[], name='labelVariable')

        return self.constantImageVar, self.constantLabelVar

    def InitializeConstantData(self, sess):
        sess.run(self.constantImageVar.initializer)
        sess.run(self.constantLabelVar.initializer)

if __name__ == '__main__':
    fileNames = ['/data/psturm/structural/structural_test.bin']
    testDataset = DataSetBIN(fileNames, imageDims=[121, 145, 121, 1], batchSize=1, maxItemsInQueue=1, minItemsInQueue=1, shuffle=False, training=False)
    bootstrapDataset = DataSetBIN(fileNames, imageDims=[121, 145, 121, 1], batchSize=1, maxItemsInQueue=1, minItemsInQueue=1, shuffle=True, training=False)
    
