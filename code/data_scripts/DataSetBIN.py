import tensorflow as tf
import numpy as np
from utils.config import get

class DataSetBIN(object):
    """
    This class creates a tensorflow queue that reads in raw byte files
    that contain 3D images with a specified dimension. 
    """
    def __init__(self,
            binFileNames,
            imageDims=[145, 145, 1],
            bytesPerNumber=8,
            decodeType=tf.float64,
            numReaderThreads=16,
            batchSize=64,
            maxItemsInQueue=2000,
            minItemsInQueue=500,
            shuffle=True,
            spliceInputAlongAxis=None,
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

        if spliceInputAlongAxis != None:
            images = tf.squeeze(self.imageBatchOperation, axis=0)
            if spliceInputAlongAxis == 0: #X AXIS
                images = tf.pad(images, [[0,0], [0,0], [24, 0], [0,0]])
            if spliceInputAlongAxis == 1: #Y AXIS
                images = tf.transpose(images, perm=[1, 0, 2, 3])
                images = tf.pad(images, [[0, 0], [24, 0], [24, 0], [0, 0]])
            elif spliceInputAlongAxis == 2: #Z AXIS
                images = tf.transpose(images, perm=[2, 0, 1, 3])
                images = tf.pad(images, [[0,0], [24, 0], [0,0], [0,0]])
            elif spliceInputAlongAxis == 3: #Hack to specify all axes
                imagesX = tf.pad(images, [[0,0], [0,0], [24, 0], [0,0]])
                imagesY = tf.pad(tf.transpose(images, perm=[1, 0, 2, 3]),
                                 [[0, 0], [24, 0], [24, 0], [0, 0]])
                imagesZ = tf.pad(tf.transpose(images, perm=[2, 0, 1, 3]),
                                 [[0,0], [24, 0], [0,0], [0,0]])
                images = tf.concat([imagesX, imagesY, imagesZ], axis=0)
            self.imageBatchOperation = images
            self.labelBatchOperation = tf.reshape(self.labelBatchOperation, shape=())

    def GetBatchOperations(self):
        return self.imageBatchOperation, self.labelBatchOperation
