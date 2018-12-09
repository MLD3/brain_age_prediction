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
            augment='none',
            augRatio=None,
            numEpochs=None
        ):
        self.filenames = filenames
        self.batchSize = batchSize
        self.imageBatchDims = imageBatchDims
        self.labelBatchDims = labelBatchDims
        self.imageBaseString = imageBaseString
        self.labelBaseString = labelBaseString
        self.maxItemsInQueue = maxItemsInQueue
        self.phenotypeBatchOperation = None
        self.preloaded = False
        self.loadedImages = None
        self.loadedLabels = None
        self.shuffle = shuffle
        stringQueue = tf.train.string_input_producer(filenames, num_epochs=numEpochs, shuffle=shuffle, capacity=maxItemsInQueue)
        self.stringQueue = stringQueue
        if not numEpochs:
            dequeueOp = stringQueue.dequeue_many(batchSize)
        else:
            dequeueOp = stringQueue.dequeue_up_to(batchSize)
        self.dequeueOp = dequeueOp
        self.imageBatchOperation = tf.reshape(
            tf.py_func(self._loadImages, [dequeueOp], tf.float32),
            imageBatchDims)
        self.labelBatchOperation = tf.reshape(
            tf.py_func(self._loadLabels, [dequeueOp], tf.float32),
            labelBatchDims)
        self.augment = augment
        if self.augment != 'none':
            self.CreateAugmentOperations(augmentation=augment, augRatio=augRatio)

    def PreloadData(self):
        files = [x.encode() for x in self.filenames]
        self.loadedImages = np.reshape(self._loadImages(files), self.imageBatchDims).astype(np.float32)
        self.loadedLabels = np.reshape(self._loadLabels(files), self.labelBatchDims).astype(np.float32)
        self.maxItemsInQueue = 1
        self.preloaded = True

    def UnloadData(self):
        self.loadedImages = None
        self.loadedLabels = None
        self.maxItemsInQueue = 100
        self.preloaded = False
        
    def NextBatch(self, sess):
        if self.preloaded:
            return self.loadedImages, self.loadedLabels

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

    def RefreshNumEpochs(self):
        '''
        This is a function that used to refresh the number of epochs.
        Only used in validation set input produce process.
        '''
        stringQueue = tf.train.string_input_producer(self.filenames, num_epochs=self.numEpochs, shuffle=self.shuffle, capacity=self.maxItemsInQueue)
        self.stringQueue = stringQueue
        if not self.numEpochs:
            dequeueOp = stringQueue.dequeue_many(self.batchSize)
        else:
            dequeueOp = stringQueue.dequeue_up_to(self.batchSize)
        self.dequeueOp = dequeueOp
        self.imageBatchOperation = tf.reshape(
            tf.py_func(self._loadImages, [dequeueOp], tf.float32),
            imageBatchDims)
        self.labelBatchOperation = tf.reshape(
            tf.py_func(self._loadLabels, [dequeueOp], tf.float32),
            labelBatchDims)

    def CreateAugmentOperations(self, augmentation='flip', augRatio=1):
        """
        These are untested features. Augmentation on the fly.
        Will choose to do augmenting operation specified by arguments with a probability of 1/2.
        Flip: flip the batch by its first axis.
        Translate: pad the images with random size of zeros, and 
                   then cut the image to remain the size.
        ---Below features are moved to elsewhere---
        Max_avg_mixture: randomly use max pooling instead of average pooling
                         in the sampling phase.
        Combination: choose two different person with the same age,
                     then combine the left and right half from each image
                     to create a new one.
        """
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
                                                sliceEnd,
                                                name='translate')
            elif augmentation == 'rotate':
                # for now it is rotate 180 degrees in x axis and y axis only
                # if proven to be useful, then we can explore more on this
                augmentedImageOperation = tf.reverse(self.imageBatchOperation,
                                                     axis=[1],
                                                     name='rotate_y')
                augmentedImageOperation = tf.reverse(augmentedImageOperation,
                                                     axis=[0],
                                                     name='rotate_x')
            # elif augmentation == 'crop':
                # TODO: what is the crop size?

            chooseOperation = tf.cond(
                tf.less(
                    tf.ones(shape=(), dtype=tf.float32),
                    tf.random_uniform(shape=(), dtype=tf.float32, minval=0, maxval=1+augRatio)
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
'''
The main function may be used for testing.
Unknown feature.
'''
if __name__ == '__main__':
    dataset = DataSetNPY(filenames=np.load('/data1/brain/ABIDE/ABIDE2/IQData/train_IQ.npy').tolist(),
                         imageBatchDims=(-1, 41, 49, 41, 1),
                         imageBaseString='/data1/brain/ABIDE/ABIDE2/avgpool3x3x3/',
                         labelBaseString='/data1/brain/ABIDE/ABIDE2/binary_labels/',
                         batchSize=5)
    imageOp, labelOp = dataset.GetBatchOperations()
    dequeueOp = dataset.dequeueOp
    dataset.CreatePhenotypicOperations(phenotypicBaseStrings=[
        '/data1/brain/ABIDE/ABIDE2/gender/',
        '/data1/brain/ABIDE/ABIDE2/IQData/FIQ/',
        '/data1/brain/ABIDE/ABIDE2/IQData/VIQ/',
        '/data1/brain/ABIDE/ABIDE2/IQData/PIQ/',
        '/data1/brain/ABIDE/ABIDE2/ages/'
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
