from keras.applications import MobileNetV2, InceptionV3, ResNet50, NASNetMobile, NASNetLarge
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Reshape, Input, Flatten, Subtract, Lambda, Concatenate
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras.callbacks import LambdaCallback, TensorBoard, LearningRateScheduler
from keras.utils import multi_gpu_model
from keras.regularizers import l1, l2
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa
import cv2
import csv
import tempfile
import tensorflow as tf
from scipy.misc import imsave, imread
import os
import sklearn.metrics
import numpy
import pickle
import psutil
import matplotlib.pyplot as plt
import os.path
import scipy.stats
import random
import copy
import sklearn.neighbors
import json
import sys
import threading
import time
from pprint import pprint
import math
import skimage.transform
import skimage.io
import random
import gc
import concurrent.futures
import multiprocessing
from tensorflow.python.client import device_lib
from keras.backend.tensorflow_backend import set_session
import datetime

class PillRecognitionModel:
    def __init__(self, parameters, generatedDataset, realDataset):
        self.parameters = parameters

        self.generatedDataset = generatedDataset
        self.realDataset = realDataset

        self.workers = int(psutil.cpu_count()*0.7)

        self.pretrainEpochs = int(parameters['startingWeights'].get('pretrainPercent', 0) * parameters['neuralNetwork']['epochs'])
        self.trainFinalLayersFirst = (parameters['startingWeights']['weights'] == 'imagenet') and self.pretrainEpochs > 0
        if self.trainFinalLayersFirst:
            self.epochs = parameters['neuralNetwork']['epochs'] - self.pretrainEpochs
        else:
            self.epochs = parameters['neuralNetwork']['epochs']

        self.enableTensorboard = parameters['enableTensorboard']

        if parameters['numGPUs'] == -1:
            self.numGPUs = len(self.getAvailableGpus())
        else:
            self.numGPUs = min(parameters['numGPUs'], len(self.getAvailableGpus()))

        self.trainingBatchNumber = multiprocessing.Value('i', 0)
        self.trainingBatchNumberLock = multiprocessing.Lock()

        self.testingBatchNumber = multiprocessing.Value('i', 0)
        self.testingBatchNumberLock = multiprocessing.Lock()

        self.imageWidth = 224
        self.imageHeight = 224

        self.testingBatchSize = int(parameters['neuralNetwork']['batchSize'] * parameters['neuralNetwork']['augmentationsPerImage'])
        self.datasetSizesToTest = parameters['datasetSizesToTest']
        self.rotationsToTest = parameters['rotationsToTest']
        self.testingPrintEvery = parameters['testingPrintEvery']
        self.measurementImages = []
        self.measurementRotatedImages = {}

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        set_session(self.session)

        self.maxImagesToGenerate = 1000
        self.imageGenerationManager = multiprocessing.Manager()
        self.imageGenerationExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=parameters['imageGenerationWorkers'])
        self.augmentedRealImages = []
        self.augmentedGeneratedImages = []
        self.imageGenerationThreads = []
        self.imageListLock = threading.Lock()
        self.augmentedTripletListLock = threading.Lock()
        self.running = False
        self.measuringAccuracy = False
        self.finished = False
        self.startTime = datetime.datetime.now()
        self.imageId = 0
        self.imageIdLock = threading.Lock()

        self.startEpoch = parameters.get('startEpoch', 0)

        if parameters['trainingAugmentation']['rotationEasing']['easing'] == 'epoch':
            self.currentMinRotation = self.imageGenerationManager.Value('i', 0)
            self.currentMaxRotation = self.imageGenerationManager.Value('i', 1)
            self.updateRotationValues(self.startEpoch)
        else:
            self.currentMinRotation = self.imageGenerationManager.Value('i', parameters['trainingAugmentation']['minRotation'])
            self.currentMaxRotation = self.imageGenerationManager.Value('i', parameters['trainingAugmentation']['maxRotation'])

        time.sleep(2)

        for n in range(parameters['imageGenerationWorkers']):
            newThread = threading.Thread(target=self.imageGenerationThread, daemon=True)
            self.imageGenerationThreads.append(newThread)

    def imageGenerationThread(self):
        while threading.main_thread().isAlive() and not self.finished:
            try:
                with self.imageIdLock:
                    imageId = self.imageId
                    self.imageId += 1

                if imageId % 100 == 0:
                    gc.collect()
                    currentImageExecutor = self.imageGenerationExecutor
                    self.imageGenerationExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=self.parameters['imageGenerationWorkers'])
                    currentImageExecutor.shutdown(wait=False)

                if imageId % 2 == 0:
                    self.generatedDataset.setRotationParams(self.currentMinRotation.value, self.currentMaxRotation.value)
                    future = self.imageGenerationExecutor.submit(self.generatedDataset.getTrainingImageSet, imageId)
                    triplet = future.result()

                    with self.augmentedTripletListLock:
                        self.augmentedGeneratedImages.append(triplet)
                        if len(self.augmentedGeneratedImages) >= self.maxImagesToGenerate:
                            del self.augmentedGeneratedImages[0]
                else:
                    self.realDataset.setRotationParams(self.currentMinRotation.value, self.currentMaxRotation.value)
                    future = self.imageGenerationExecutor.submit(self.realDataset.getTrainingImageSet, imageId)
                    triplet = future.result()

                    with self.augmentedTripletListLock:
                        self.augmentedRealImages.append(triplet)
                        if len(self.augmentedRealImages) >= self.maxImagesToGenerate:
                            del self.augmentedRealImages[0]
                if not self.running or self.measuringAccuracy:
                    time.sleep(1.0)
            except OSError as e:
                time.sleep(2.0)
            except RuntimeError as e:
                time.sleep(2.0)

    def getAvailableGpus(self):
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def createTripletLoss(self):
        """
        :param log_loss: Whether or not the loss should be passed through a log function
        :param sum_loss: Whether the loss function should be applied seperately to each vector, or to the sum of the whole vector
        :return:
        """
        pass


        def triplet_loss(y_true, y_pred, epsilon=1e-6):
            """
            Implementation of the triplet loss function

            Arguments:
            y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
            y_pred -- python list containing three objects:
                    anchor -- the encodings for the anchor data
                    positive -- the encodings for the positive data (similar to anchor)
                    negative -- the encodings for the negative data (different from anchor)
            epsilon -- The Epsilon value to prevent ln(0)


            Returns:
            loss -- real number, value of the loss
            """
            losses = []
            for n in range(int(self.parameters['neuralNetwork']['batchSize'])):
                for k in range(self.parameters['neuralNetwork']['augmentationsPerImage']):
                    anchor = tf.convert_to_tensor(y_pred[n*self.parameters['neuralNetwork']['augmentationsPerImage'] + k])

                    posDists = []
                    negDists = []

                    for d in range(self.parameters['neuralNetwork']['augmentationsPerImage']):
                        if d != k:
                            positive = tf.convert_to_tensor(y_pred[n*self.parameters['neuralNetwork']['augmentationsPerImage'] + d])
                            pos_diff = tf.square(tf.subtract(anchor, positive))
                            # distance between the anchor and the positive
                            pos_dist = tf.reduce_sum(pos_diff)
                            posDists.append(pos_dist)

                    for j in range(int(self.parameters['neuralNetwork']['batchSize'])):
                        if j != n:
                            for m in range(self.parameters['neuralNetwork']['augmentationsPerImage']):
                                negative = tf.convert_to_tensor(y_pred[j * self.parameters['neuralNetwork']['augmentationsPerImage'] + m])
                                neg_diff = tf.square(tf.subtract(anchor, negative))
                                # distance between the anchor and the negative
                                neg_dist = tf.reduce_sum(neg_diff)
                                negDists.append(neg_dist)

                    if self.parameters['neuralNetwork']['lossMode'] == 'hard':
                        worstPosDist = tf.reduce_max(tf.stack(posDists))
                        worstNegDist = tf.reduce_min(tf.stack(negDists))
                        pos_dist = tf.divide((worstPosDist), self.parameters['neuralNetwork']['vectorSize'])
                        neg_dist = tf.divide((worstNegDist), self.parameters['neuralNetwork']['vectorSize'])
                        loss = tf.maximum(pos_dist - neg_dist + self.parameters['neuralNetwork']['lossMargin'], 0.0)
                        losses.append(loss)
                    elif self.parameters['neuralNetwork']['lossMode'] == 'all':
                        for pos in posDists:
                            for neg in negDists:
                                pos_dist = tf.divide((pos), self.parameters['neuralNetwork']['vectorSize'])
                                neg_dist = tf.divide((neg), self.parameters['neuralNetwork']['vectorSize'])
                                loss = tf.maximum(pos_dist - neg_dist + self.parameters['neuralNetwork']['lossMargin'], 0.0)
                                losses.append(loss)

            tf.summary.tensor_summary(
                "batch_nonzero",
                tf.count_nonzero(losses),
                summary_description=None,
                collections=None,
                summary_metadata=None,
                family=None,
                display_name=None
            )

            # regularizationLosses = []
            # for n in range(int(self.parameters['neuralNetwork']['batchSize'])):
            #     for k in range(self.parameters['neuralNetwork']['augmentationsPerImage']):
            #         anchor = tf.convert_to_tensor(y_pred[n*self.parameters['neuralNetwork']['augmentationsPerImage'] + k])
            #         anchorSorted = tf.contrib.framework.sort(anchor)
            #         penalty = tf.reduce_mean(anchorSorted[0:int(self.parameters['neuralNetwork']['vectorSize']/2)]) \
            #         - tf.reduce_mean(anchorSorted[int(self.parameters['neuralNetwork']['vectorSize']/2):])
            #
            #         regularizationLosses.append(penalty)

            return tf.reduce_mean(losses) # + tf.reduce_mean(regularizationLosses) * 0.01
        return triplet_loss

    def generateBatch(self, testing=False):
        batchNumber = 0
        while True:
            batchNumber += 1

            # Generate half the batch as negative examples, half the batch as positive examples
            inputs = []
            outputs = []

            if batchNumber % 2 == 0 and batchNumber > 500:
                triplets = random.sample(range(min(len(self.augmentedRealImages)-1, self.maxImagesToGenerate-1)), min(len(self.augmentedRealImages)-1, int(self.parameters['neuralNetwork']['batchSize'])))
                augmentedImages = self.augmentedRealImages
            else:
                triplets = random.sample(range(min(len(self.augmentedGeneratedImages)-1, self.maxImagesToGenerate-1)), min(len(self.augmentedGeneratedImages)-1, int(self.parameters['neuralNetwork']['batchSize'])))
                augmentedImages = self.augmentedGeneratedImages

            # Add each image into the batch
            for n in range(int(self.parameters['neuralNetwork']['batchSize'])):
                tripletInputs = augmentedImages[triplets[n]]

                for input in range(self.parameters['neuralNetwork']['augmentationsPerImage']):
                    inputs.append(tripletInputs[input])
                    outputs.append(numpy.ones(self.parameters['neuralNetwork']['vectorSize']))

            # Add in blends to the batch
            inputs = numpy.array(inputs)
            outputs = numpy.array(outputs)

            yield inputs, outputs

    def loadModel(self, fileName):
        self.model, imageNet = self.createCoreModel()
        imageNet.load_weights(fileName)

    def createCoreModel(self):
        primaryDevice = "/cpu:0"
        if self.numGPUs == 1:
            primaryDevice = "/gpu:0"

        with tf.device(primaryDevice):
            imageNet = Sequential()
            imageNetCore = None
            if self.parameters['neuralNetwork']['core'] == 'resnet':
                imageNetCore = ResNet50(include_top=False, pooling='avg', input_shape=(self.imageWidth, self.imageHeight, 3), weights=('imagenet' if self.parameters['startingWeights']['weights'] == 'imagenet' else None))
            elif self.parameters['neuralNetwork']['core'] == 'inceptionnet':
                imageNetCore = InceptionV3(include_top=False, pooling='avg', input_shape=(self.imageWidth, self.imageHeight, 3), weights=('imagenet' if self.parameters['startingWeights']['weights'] == 'imagenet' else None))
            elif self.parameters['neuralNetwork']['core'] == 'mobilenet':
                imageNetCore = MobileNetV2(include_top=False, pooling='avg', input_shape=(self.imageWidth, self.imageHeight, 3), weights=('imagenet' if self.parameters['startingWeights']['weights'] == 'imagenet' else None))
            elif self.parameters['neuralNetwork']['core'] == 'nasnetmobile':
                imageNetCore = NASNetMobile(include_top=False, pooling='avg', input_shape=(self.imageWidth, self.imageHeight, 3), weights=('imagenet' if self.parameters['startingWeights']['weights'] == 'imagenet' else None))
            elif self.parameters['neuralNetwork']['core'] == 'nasnetlarge':
                imageNetCore = NASNetLarge(include_top=False, pooling='avg', input_shape=(self.imageWidth, self.imageHeight, 3), weights=('imagenet' if self.parameters['startingWeights']['weights'] == 'imagenet' else None))

            imageNetCore.summary()

            imageNet.add(imageNetCore)
            imageNet.add(Reshape([2048]))
            imageNet.add(BatchNormalization())
            imageNet.add(Dropout(self.parameters["neuralNetwork"]["dropoutRate"]))
            imageNet.add(Dense(int(self.parameters['neuralNetwork']['vectorSize']*self.parameters["neuralNetwork"]["denseLayerMultiplier"])))
            imageNet.add(BatchNormalization())
            # imageNet.add(Dense(int(self.parameters['neuralNetwork']['vectorSize'])))
            imageNet.add(Dense(int(self.parameters['neuralNetwork']['vectorSize']), activation=self.parameters["neuralNetwork"]["finalActivation"]))
            # imageNet.add(Dense(int(self.parameters['neuralNetwork']['vectorSize']), activation=self.parameters["neuralNetwork"]["finalActivation"]))

            imageNet.summary()

        if self.numGPUs > 1:
            model = multi_gpu_model(imageNet, gpus=self.numGPUs)
        else:
            model = imageNet
        
        return model, imageNet

    def updateRotationValues(self, epoch):
        self.currentMinRotation.value = min(1.0, float(epoch+1) / float(
            (self.parameters['trainingAugmentation']['rotationEasing']['minRotationEasing'] * self.parameters['neuralNetwork']['epochs']))) * self.parameters['trainingAugmentation']['minRotation']
        self.currentMaxRotation.value = min(1.0, float(epoch+1) / float(
            (self.parameters['trainingAugmentation']['rotationEasing']['maxRotationEasing'] * self.parameters['neuralNetwork']['epochs']))) * self.parameters['trainingAugmentation']['maxRotation']

    def learningRateForEpoch(self, epoch):
        return self.parameters['neuralNetwork']['optimizer']['learningRate'] * (self.parameters['neuralNetwork']['optimizer']['learningRateDecay'] ** epoch)

    def trainModel(self):
        for thread in self.imageGenerationThreads:
            thread.start()

        self.model, imageNet = self.createCoreModel()

        testingGenerator = self.generateBatch(testing=True)
        trainingGenerator = self.generateBatch(testing=False)

        bestAccuracy = None
        allAccuracies = []

        def epochCallback(epoch, logs):
            nonlocal bestAccuracy
            self.measuringAccuracy=True
            if self.parameters['trainingAugmentation']['rotationEasing']['easing'] == 'epoch':
                self.updateRotationValues(epoch)

            if epoch % 5 == 0:
                imageNet.save(f"model-epoch-{epoch}.h5")
                imageNet.save_weights(f"model-epoch-{epoch}-weights.h5")
            imageNet.save(f"model-current.h5")
            imageNet.save_weights(f"model-current-weights.h5")

            if epoch % self.parameters['epochsBeforeAccuracyMeasurement'] == (self.parameters['epochsBeforeAccuracyMeasurement']-1):
                accuracy = self.measureAccuracy(self.model)
                if bestAccuracy is None or accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    imageNet.save_weights(f"model-best-weights.h5")
                allAccuracies.append(accuracy)
            self.measuringAccuracy = False

        rollingAverage9 = None
        rollingAverage95 = None
        rollingAverage99 = None
        rollingAverage995 = None
        def batchCallback(batch, log):
            nonlocal rollingAverage9, rollingAverage95, rollingAverage99, rollingAverage995
            # if batch % 100 == 0:
            #     self.memory_tracker.print_diff()
            if rollingAverage9 is None:
                rollingAverage95 = log['loss']
                rollingAverage9 = log['loss']
                rollingAverage99 = log['loss']
                rollingAverage995 = log['loss']

            rollingAverage9 = log['loss'] * 0.1 + rollingAverage9 * 0.9
            rollingAverage95 = log['loss'] * 0.05 + rollingAverage95 * 0.95
            rollingAverage99 = log['loss'] * 0.01 + rollingAverage99 * 0.99
            rollingAverage995 = log['loss'] * 0.005 + rollingAverage995 * 0.995

            trend95 = '+' if rollingAverage9 > rollingAverage95 else '-'
            trend99 = '+' if rollingAverage95 > rollingAverage99 else '-'
            trend995 = '+' if rollingAverage99 > rollingAverage995 else '-'
            trend = trend95 + trend99 + trend995

            print("  batch loss", log['loss'], "  rl9  ", rollingAverage9,  "  rl99", rollingAverage99, "  trend ", trend)

        testNearestNeighbor = LambdaCallback(on_epoch_end=epochCallback, on_batch_end=batchCallback)

        learningRateScheduler = LearningRateScheduler(lambda epoch, lr: self.learningRateForEpoch(epoch))

        callbacks = [testNearestNeighbor, learningRateScheduler]
        optimizer = None

        if 'loadFile' in self.parameters:
            imageNet.load_weights(self.parameters['loadFile'])

        if self.enableTensorboard:
            tensorBoardCallback = CustomTensorBoard(
                user_defined_freq=1,
                log_dir='./logs',
                histogram_freq=5,
                batch_size=self.parameters['neuralNetwork']['batchSize'],
                write_graph=True,
                write_grads=False,
                write_images=False,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None,
                embeddings_data=None,
                update_freq='batch')
            callbacks.append(tensorBoardCallback)

        if self.trainFinalLayersFirst:
            imageNet.layers[0].trainable = False

            if self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'adam':
                optimizer = Adam(self.learningRateForEpoch(self.startEpoch))
            elif self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'nadam':
                optimizer = Nadam(self.learningRateForEpoch(self.startEpoch))
            elif self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'rmsprop':
                optimizer = RMSprop(self.learningRateForEpoch(self.startEpoch))
            elif self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'sgd':
                optimizer = SGD(self.learningRateForEpoch(self.startEpoch))

            self.model.compile(loss=self.createTripletLoss(), optimizer=optimizer)

            self.model.summary()
            self.model.count_params()
            self.running = True

            self.model.fit_generator(
                generator=trainingGenerator,
                steps_per_epoch=self.parameters['stepsPerEpoch'],
                epochs=self.pretrainEpochs,
                validation_data=testingGenerator,
                validation_steps=self.parameters['validationSteps'],
                workers=1,
                use_multiprocessing=False,
                max_queue_size=self.parameters['maxQueueSize'],
                callbacks=callbacks,
                initial_epoch=self.startEpoch
            )

            imageNet.layers[0].trainable = True

        if self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'adam':
            optimizer = Adam(self.learningRateForEpoch(self.startEpoch))
        elif self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'nadam':
            optimizer = Nadam(self.learningRateForEpoch(self.startEpoch))
        elif self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'rmsprop':
            optimizer = RMSprop(self.learningRateForEpoch(self.startEpoch))
        elif self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'sgd':
            optimizer = SGD(self.learningRateForEpoch(self.startEpoch))


        self.model.compile(loss=self.createTripletLoss(), optimizer=optimizer)

        self.model.summary()
        self.model.count_params()
        self.running = True

        currentEpoch = self.startEpoch
        while currentEpoch < self.epochs:
            self.model.fit_generator(
                generator=trainingGenerator,
                steps_per_epoch=self.parameters['stepsPerEpoch'],
                epochs=self.epochs,
                validation_data=testingGenerator,
                validation_steps=self.parameters['validationSteps'],
                workers=1,
                use_multiprocessing=False,
                max_queue_size=self.parameters['maxQueueSize'],
                callbacks=callbacks,
                initial_epoch=currentEpoch
            )
            currentEpoch += 1

        imageNet.save(f"model-final.h5")
        imageNet.save_weights(f"model-final-weights.h5")

        self.finished = True
        time.sleep(5)
        self.imageGenerationExecutor.shutdown()
        del self.imageGenerationExecutor
        K.clear_session()
        self.session.close()
        del self.session
        time.sleep(5)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(list(range(len(allAccuracies))), allAccuracies)

        return min(1, max(0, slope * 20))

    def measureAccuracy(self, model):
        print("Measuring Accuracy", flush=True)
        maxDatasetSize = max(*self.datasetSizesToTest)

        if len(self.measurementImages) < maxDatasetSize:
            self.measurementRotatedImages = {
                rotation: [] for rotation in self.rotationsToTest
            }

            print("    Generating test images", flush=True)
            imageId = 0
            completedImages = 0
            # Build images in sets of 20 batches. This is to get around a python multiprocessing bug.
            for k in range(25):
                futures = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as worker:
                    for n in range(int(maxDatasetSize/25)):
                        futures.append(worker.submit(self.realDataset.getRotationTestingImageSet, imageId))
                        imageId += 1

                    for future in concurrent.futures.as_completed(futures):
                        self.measurementImages.append(future.result()[0])
                        for rotation in self.rotationsToTest:
                            self.measurementRotatedImages[rotation].append(future.result()[1][rotation])
                        completedImages += 1

                        if completedImages % self.testingPrintEvery == 0:
                            print(f"        Completed generating {completedImages} images.", flush=True)

        originalVectors = {
            rotation: [] for rotation in self.rotationsToTest
        }
        rotatedVectors = {
            rotation: [] for rotation in self.rotationsToTest
        }

        print("    Computing vectors for images", flush=True)
        for batch in range(int(math.ceil(maxDatasetSize / self.testingBatchSize))):
            batchStart = batch * self.testingBatchSize
            batchEnd = (batch + 1) * self.testingBatchSize


            batchOriginals = self.measurementImages[batchStart:batchEnd]
            if len(batchOriginals) == 0:
                break
            batchOriginalVectors = model.predict(numpy.array(batchOriginals))

            for rotation in self.rotationsToTest:
                batchRotated = self.measurementRotatedImages[rotation][batchStart:batchEnd]
                batchRotatedVectors = model.predict(numpy.array(batchRotated))

                for n in range(len(batchOriginals)):
                    originalVector = batchOriginalVectors[n]
                    rotatedVector = batchRotatedVectors[n]

                    if numpy.any(numpy.isnan(originalVector)) or numpy.any(numpy.isnan(rotatedVector)) or numpy.any(numpy.isinf(originalVector)) or numpy.any(numpy.isinf(rotatedVector)):
                        print("        Error, NaN or infinity in final vector", flush=True)
                    else:
                        originalVectors[rotation].append(originalVector)
                        rotatedVectors[rotation].append(rotatedVector)

                        if rotation == self.rotationsToTest[0] and len(originalVectors[rotation]) % self.testingPrintEvery == 0:
                            print(f"        Completed vectors for {len(originalVectors[rotation])} samples", flush=True)

        print("    Measuring Final Accuracy", flush=True)
        top1Accuracies = []
        accuracyRows = []
        for rotation in self.rotationsToTest:
            if len(originalVectors[rotation]) == 0:
                print(f"        Error! Unable to measure accuracy for rotation {rotation}. All samples had bad vectors.", flush=True)
                continue

            for datasetSize in self.datasetSizesToTest:
                print(f"        Measuring accuracy on {rotation} degree rotations with dataset size {datasetSize}", flush=True)

                vectorIndexes = random.sample(range(len(originalVectors[rotation])), min(len(originalVectors[rotation]), datasetSize))

                origVectorsForTest = [originalVectors[rotation][index] for index in vectorIndexes]
                rotatedVectorsForTest = [rotatedVectors[rotation][index] for index in vectorIndexes]

                print(f"            Fitting the nearest neighbor model on {len(origVectorsForTest)} samples", flush=True)
                nearestNeighborModel = sklearn.neighbors.NearestNeighbors(n_neighbors=5)
                nearestNeighborModel.fit(origVectorsForTest)

                print(f"            Computing results of nearest neighbors model on rotated images.", flush=True)
                distance, indices = nearestNeighborModel.kneighbors(rotatedVectorsForTest)

                correct = {topk: 0 for topk in range(1, 6)}
                for n in range(len(origVectorsForTest)):
                    for topk in range(1, 6):
                        for k in range(0, topk):
                            if indices[n][k] == n:
                                correct[topk] += 1
                                break

                for topk in range(1, 6):
                    accuracy = float(correct[topk]) / float(len(origVectorsForTest))
                    print(f"            Nearest Neighbor Top-{topk} Accuracy on {rotation} degree rotations with {datasetSize} total dataset size: {accuracy}", flush=True)

                    if topk == 1:
                        top1Accuracies.append(accuracy)

                    accuracyRows.append({
                        "rotation": rotation,
                        "topk": topk,
                        "accuracy": accuracy,
                        "datasetSize": datasetSize
                    })

        if len(top1Accuracies) == 0:
            return 0

        writer = csv.DictWriter(sys.stdout, fieldnames=list(accuracyRows[0].keys()), dialect=csv.excel_tab)
        writer.writeheader()
        writer.writerows(accuracyRows)
        print("")
        meanTop1Accuracy = numpy.mean(top1Accuracies)
        print(f"    Final Mean Top-1 Accuracy {meanTop1Accuracy}", flush=True)
        return meanTop1Accuracy


    def finalMeasureAccuracy(self, model):
        print("Measuring Accuracy", flush=True)
        dbVectors = {}
        testVectors = {}
        currentImageId = 0

        if not os.path.exists('data'):
            os.mkdir("data")
        if not os.path.exists("data/images"):
            os.mkdir("data/images")

        vectorsFile = open('vectors.tsv', 'wt')
        vectorsCSVWriter = csv.DictWriter(vectorsFile, fieldnames=list(range(self.parameters['neuralNetwork']['vectorSize'])), dialect=csv.excel_tab)

        metadataFile = open('metadata.tsv', 'wt')
        metadataCSVWriter = csv.DictWriter(metadataFile, fieldnames=['imageId', 'augmentationId', 'index'], dialect=csv.excel_tab)
        metadataCSVWriter.writeheader()

        print("    Generating and vectorizing test images", flush=True)
        completedImages = 0
        # Build images in sets of 10 batches. This is to get around a python multiprocessing bug.
        with tempfile.TemporaryDirectory() as dir:
            for n in range(int(self.parameters['finalTestImages'])):
                imageId = currentImageId
                currentImageId += 1

                result = self.generatedDataset.getFinalTestingImageSet(imageId)
                
                imageId = result[0]
                dbImages = result[1]
                testImages = result[2]

                if not os.path.exists(f"data/images/{imageId}"):
                    os.mkdir(f"data/images/{imageId}")

                for imageIndex, image in enumerate(dbImages):
                    skimage.io.imsave(f"data/images/{imageId}/db-{imageIndex}.png", image)

                for augmentationIndex, augmentation in enumerate(testImages):
                    for imageIndex, image in enumerate(augmentation):
                        skimage.io.imsave(f"data/images/{imageId}/test-{augmentationIndex}-{imageIndex}.png", image)

                dbVectors[imageId] = sklearn.preprocessing.normalize(model.predict(numpy.array(dbImages)))

                for vectorIndex, vector in enumerate(dbVectors[imageId]):
                    metadataCSVWriter.writerow({
                        "imageId": imageId,
                        "augmentationId": "db",
                        "index": vectorIndex
                    })
                    vectorsCSVWriter.writerow({index: value for index,value in enumerate(vector)})

                testVectors[imageId] = []
                for augmentationIndex, augmentation in enumerate(testImages):
                    vectors = sklearn.preprocessing.normalize(model.predict(numpy.array(augmentation)))
                    testVectors[imageId].append(vectors)

                    for vectorIndex, vector in enumerate(vectors):
                        metadataCSVWriter.writerow({
                            "imageId": imageId,
                            "augmentationId": augmentationIndex,
                            "index": vectorIndex
                        })
                        vectorsCSVWriter.writerow({index: value for index,value in enumerate(vector)})

                completedImages += 1

                if completedImages % self.parameters['finalTestPrintEvery'] == 0:
                    print(f"        Completed {completedImages} images.", flush=True)

        vectorsFile.close()
        metadataFile.close()

        print("    Measuring Final Accuracy", flush=True)


        concatDBVectors = []
        concatDBIds = []
        for imageId in dbVectors:
            for vector in dbVectors[imageId]:
                concatDBVectors.append(vector)
                concatDBIds.append(imageId)

        print("Fitting model")
        nearestNeighborModel = sklearn.neighbors.NearestNeighbors(n_neighbors=30)
        nearestNeighborModel.fit(concatDBVectors)

        for neigbors in range(1, 30, 1):
            print("Computing match results for neighbors ", neigbors)
            totalCorrectCountMethod = 0
            totalCorrectMinDistMethod = 0
            totalCorrectAvgDistMethod = 0
            total = 0
            for imageId in testVectors:
                for augIndex, augmentation in enumerate(testVectors[imageId]):
                    rotationDistances, rotationMatchingIndexes = nearestNeighborModel.kneighbors(augmentation, n_neighbors=neigbors)

                    totalCounts = {}
                    totalDistances = {}
                    minDistances = {}
                    for distances, indexes in zip(rotationDistances, rotationMatchingIndexes):
                        for distance, index in zip(distances, indexes):
                            predictId = concatDBIds[index]
                            totalCounts[predictId] = totalCounts.get(predictId, 0) + 1
                            totalDistances[predictId] = totalDistances.get(predictId, 0) + distance
                            minDistances[predictId] = min(minDistances.get(predictId, distance), distance)

                    maxCount = None
                    maxCountId = None

                    minDist = None
                    minDistId = None

                    minAvgDist = None
                    minAvgDistId = None

                    for predictId in totalCounts:
                        if maxCount is None or totalCounts[predictId] > maxCount:
                            maxCount = totalCounts[predictId]
                            maxCountId = predictId

                        if minDist is None or minDistances[predictId] < minDist:
                            minDist = minDistances[predictId]
                            minDistId = predictId

                        averageDist = totalDistances[predictId] / totalCounts[predictId]
                        if minAvgDist is None or averageDist < minAvgDist:
                            minAvgDist = averageDist
                            minAvgDistId = predictId

                    total += 1
                    if maxCountId == imageId:
                        totalCorrectCountMethod += 1
                    else:
                        print(f"imageId: {imageId}, augmentation: {augIndex}")
                        pprint(totalCounts)

                    if minDistId == imageId:
                        totalCorrectMinDistMethod += 1

                    if minAvgDistId == imageId:
                        totalCorrectAvgDistMethod += 1

                    countMethodAccuracy = float(totalCorrectCountMethod) / float(total)
                    minDistMethodAccuracy = float(totalCorrectMinDistMethod) / float(total)
                    avgDistMethodAccuracy = float(totalCorrectAvgDistMethod) / float(total)

                    if total % 100 == 0:
                        print(countMethodAccuracy, minDistMethodAccuracy, avgDistMethodAccuracy, flush=True)

            countMethodAccuracy = float(totalCorrectCountMethod) / float(total)
            minDistMethodAccuracy = float(totalCorrectMinDistMethod) / float(total)
            avgDistMethodAccuracy = float(totalCorrectAvgDistMethod) / float(total)

            print(f"Accuracy with neighbors {neigbors} is count: {countMethodAccuracy}, minDist: {minDistMethodAccuracy}, avgDist: {avgDistMethodAccuracy}", flush=True)

        return countMethodAccuracy



class CustomTensorBoard(TensorBoard):
  """Extends the TensorBoard callback to allow adding custom summaries.


  Arguments:
      user_defined_freq: frequency (in epochs) at which to compute summaries
          defined by the user by calling tf.summary in the model code. If set to
          0, user-defined summaries won't be computed. Validation data must be
          specified for summary visualization.
      kwargs: Passed to tf.keras.callbacks.TensorBoard.
  """


  def __init__(self, user_defined_freq=0, **kwargs):
    self.user_defined_freq = user_defined_freq
    super(CustomTensorBoard, self).__init__(**kwargs)


  def on_epoch_begin(self, epoch, logs=None):
    """Add user-def. op to Model eval_function callbacks, reset batch count."""


    # check if histogram summary should be run for this epoch
    if self.user_defined_freq and epoch % self.user_defined_freq == 0:
      self._epoch = epoch
      # pylint: disable=protected-access
      # add the user-defined summary ops if it should run this epoch
      self.model._make_eval_function()
      if self.merged not in self.model._eval_function.fetches:
        self.model._eval_function.fetches.append(self.merged)
        self.model._eval_function.fetch_callbacks[
            self.merged] = self._fetch_callback
      # pylint: enable=protected-access


    super(CustomTensorBoard, self).on_epoch_begin(epoch, logs=None)


  def on_epoch_end(self, epoch, logs=None):
    """Checks if summary ops should run next epoch, logs scalar summaries."""


    # pop the user-defined summary op after each epoch
    if self.user_defined_freq:
      # pylint: disable=protected-access
      if self.merged in self.model._eval_function.fetches:
        self.model._eval_function.fetches.remove(self.merged)
      if self.merged in self.model._eval_function.fetch_callbacks:
        self.model._eval_function.fetch_callbacks.pop(self.merged)
      # pylint: enable=protected-access


    super(CustomTensorBoard, self).on_epoch_end(epoch, logs=logs)
