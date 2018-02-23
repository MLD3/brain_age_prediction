import tensorflow as tf
import numpy as np
from utils.config import get

class DataSetNPY(object):
    def __init__(self,
            filenames,
            imageBaseString,
            imageBatchDims,
            labelBatchDims=(-1,1),
            labelBaseString=get('DATA.LABELS'),
            batchSize=64,
            maxItemsInQueue=100,
            shuffle=True,
            augment='none'
        ):
        self.filenames = filenames
        self.batchSize = batchSize
        self.imageBatchDims = imageBatchDims
        self.labelBatchDims = labelBatchDims
        self.imageBaseString = imageBaseString
        self.labelBaseString = labelBaseString
        stringQueue = tf.train.string_input_producer(filenames, shuffle=shuffle, capacity=maxItemsInQueue)
        dequeueOp = stringQueue.dequeue_many(batchSize)
        self.imageBatchOperation = tf.reshape(
            tf.py_func(self._loadImages, [dequeueOp], tf.float32),
            imageBatchDims)
        self.labelBatchOperation = tf.reshape(
            tf.py_func(self._loadLabels, [dequeueOp], tf.float32),
            labelBatchDims)
        self.augment = augment
        if self.augment != 'none':
            self.CreateAugmentOperations(augmentation=augment)

    def NextBatch(self, sess):
        if self.augment == 'none':
            return sess.run([self.imageBatchOperation, self.labelBatchOperation])
        else:
            return sess.run([self.augmentedImageOperation, self.labelBatchOperation])

    def GetBatchOperations(self):
        return self.imageBatchOperation, self.labelBatchOperation

    def GetRandomBatchOperations(self):
        randomIndexOperation = tf.random_uniform(shape=(self.batchSize,),
                                                dtype=tf.int32,
                                                minval=0,
                                                maxval=len(self.filenames))
        filenameTensor = tf.constant(self.filenames, dtype=tf.string)
        randomFilenames = tf.gather(filenameTensor, randomIndexOperation)
        randomImageBatch = tf.reshape(
            tf.py_func(self._loadImages, [randomFilenames], tf.float32),
            self.imageBatchDims)
        randomLabelBatch = tf.reshape(
            tf.py_func(self._loadLabels, [randomFilenames], tf.float32),
            self.labelBatchDims)
        return randomImageBatch, randomLabelBatch

    def CreateAugmentOperations(self, augmentation='flip'):
        with tf.variable_scope('DataAugmentation'):
            if augmentation == 'flip':
                augmentedImageOperation = tf.reverse(self.imageBatchOperation,
                                                     axis=[1],
                                                     name='flip')
            elif augmentation == 'translate':
                imageRank = 3
                maxPad = 6
                minPad = 0
                randomPadding = tf.random_uniform(shape=(3, 2),
                                                  minval=minPad,
                                                  maxval=maxPad + 1,
                                                  dtype=tf.int32)
                randomPadding = tf.pad(randomPadding, paddings=[[1, 1], [0, 0]])
                paddedImageOperation = tf.pad(self.imageBatchOperation, randomPadding)
                sliceBegin = randomPadding[:, 1]
                sliceEnd = self.imageBatchDims
                augmentedImageOperation = tf.slice(paddedImageOperation,
                                                sliceBegin,
                                                sliceEnd)

            chooseOperation = tf.cond(
                tf.equal(
                    tf.ones(shape=(), dtype=tf.int32),
                    tf.random_uniform(shape=(), dtype=tf.int32, minval=0, maxval=2)
                ),
                lambda: augmentedImageOperation,
                lambda: self.imageBatchOperation,
                name='ChooseAugmentation'
            )
            self.augmentedImageOperation = tf.reshape(chooseOperation, self.imageBatchDims)

    def _loadImages(self, x):
        images = []
        for name in x:
            images.append(np.load('{}{}.npy'.format(self.imageBaseString, name.decode('utf-8'))).astype(np.float32))
        images = np.array(images)
        return images

    def _loadLabels(self, x):
        labels = []
        for name in x:
            labels.append(np.load('{}{}.npy'.format(self.labelBaseString, name.decode('utf-8'))).astype(np.float32))
        labels = np.array(labels)
        return labels

if __name__ == '__main__':
    dataset = DataSetNPY(filenames=['{}'.format(i) for i in range(10)], imageBaseString='../train', labelBaseString='../label', batchSize=5)
    imageOp, labelOp = dataset.GetBatchOperations()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(10):
            images, labels = sess.run([imageOp, labelOp])
            print(images)
            print(labels)
        coord.request_stop()
        coord.join(threads)
