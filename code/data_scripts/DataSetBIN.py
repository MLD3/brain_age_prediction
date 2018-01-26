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
            shuffle=True
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

    def NextBatch(self, sess):
        """
        Returns the next batch of examples using sess. Assumes that
        tf.train.start_queue_runners has already been called. If
        it hasn't been called, this code will not execute properly.
        """
        return sess.run([self.imageBatchOperation, self.labelBatchOperation])
