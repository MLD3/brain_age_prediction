import tensorflow as tf
import numpy as np
from utils.config import get

class DataSetNPY(object):
    """
    This class supports reading in batches of .npy data.
    It is less efficient than converting your data into TFRecords (but
    it is difficult to convert 3D data into such a format), and it is also
    less efficient than converting your data into byte format
    (which is possible for 3D data, but runs into computational issues
    regarding model building). 
    """
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
        self.maxItemsInQueue = maxItemsInQueue
        self.phenotypeBatchOperation = None
        stringQueue = tf.train.string_input_producer(filenames, shuffle=shuffle, capacity=maxItemsInQueue)
        dequeueOp = stringQueue.dequeue_many(batchSize)
        self.dequeueOp = dequeueOp
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
            if self.phenotypeBatchOperation is not None:
                return sess.run([self.imageBatchOperation, self.labelBatchOperation, self.phenotypeBatchOperation])
            else:
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

    def CreatePhenotypicOperations(self, phenotypicBaseStrings):
        self.phenotypicBaseStrings = phenotypicBaseStrings
        self.phenotypeBatchOperation = tf.reshape(
            tf.py_func(self._loadPhenotypes, [self.dequeueOp], tf.float32),
            (self.batchSize, len(self.phenotypicBaseStrings) + 1))

    def _loadPhenotypes(self, x):
        #NOTE: ASSUMES THE FIRST PHENOTYPE IS GENDER,
        #WHERE MALE IS 1 AND FEMALE IS 2
        phenotypes = np.zeros((self.batchSize, len(self.phenotypicBaseStrings) + 1), dtype=np.float32)
        batchIndex = 0
        for name in x:
            phenotypeIndex = 0
            for baseString in self.phenotypicBaseStrings:
                if phenotypeIndex == 0:
                    gender = np.load('{}{}.npy'.format(baseString, name.decode('utf-8'))).astype(np.float32)
                    if gender == 1:
                        phenotypes[batchIndex, phenotypeIndex] = 1
                    elif gender == 2:
                        phenotypes[batchIndex, phenotypeIndex + 1] = 1
                    phenotypeIndex += 1
                else:
                    phenotypes[batchIndex, phenotypeIndex] = np.load('{}{}.npy'.format(baseString, name.decode('utf-8'))).astype(np.float32)

                phenotypeIndex += 1
            batchIndex += 1
        return phenotypes

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
    dataset = DataSetNPY(filenames=np.load('/data/psturm/ABIDE/ABIDE2/IQData/train_IQ.npy').tolist(),
                         imageBatchDims=(-1, 41, 49, 41, 1),
                         imageBaseString='/data/psturm/ABIDE/ABIDE2/avgpool3x3x3/',
                         labelBaseString='/data/psturm/ABIDE/ABIDE2/binary_labels/',
                         batchSize=5)
    imageOp, labelOp = dataset.GetBatchOperations()
    dequeueOp = dataset.dequeueOp
    dataset.CreatePhenotypicOperations(phenotypicBaseStrings=[
        '/data/psturm/ABIDE/ABIDE2/gender/',
        '/data/psturm/ABIDE/ABIDE2/IQData/FIQ/',
        '/data/psturm/ABIDE/ABIDE2/IQData/VIQ/',
        '/data/psturm/ABIDE/ABIDE2/IQData/PIQ/',
        '/data/psturm/ABIDE/ABIDE2/ages/'
    ])
    phenotypeOp = dataset.phenotypeBatchOperation
    config  = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1):
            subjects, images, labels, phenotypes = sess.run([dequeueOp, imageOp, labelOp, phenotypeOp])
            print(subjects)
            #print(images)
            print(labels)
            print(phenotypes)
        coord.request_stop()
        coord.join(threads)
