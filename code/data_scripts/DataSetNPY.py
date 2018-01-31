import tensorflow as tf
import numpy as np
from utils.config import get

class DataSetNPY(object):
    def __init__(self,
            filenames,
            imageBaseString,
            imageBatchDims,
            labelBatchDims=(-1,1)
            labelBaseString=get('DATA.LABELS'),
            batchSize=64,
            maxItemsInQueue=100,
            shuffle=True
        ):
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

    def GetBatchOperations(self):
        return self.imageBatchOperation, self.labelBatchOperation

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
