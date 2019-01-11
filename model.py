from keras.applications import MobileNetV2, InceptionV3, ResNet50, NASNetMobile, NASNetLarge
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Reshape, Input, merge, Flatten, Subtract, Lambda, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras.callbacks import LambdaCallback, TensorBoard, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.regularizers import l1, l2
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa
import csv
import tensorflow as tf
from scipy.misc import imsave, imread
import os
import sklearn.metrics
import numpy
import psutil
import matplotlib.pyplot as plt
import os.path
import random
import copy
import sklearn.neighbors
import json
import sys
import threading
import time
from pprint import pprint
from generate_data import generatePillImage
import math
import skimage.transform
import random
import concurrent.futures
import multiprocessing
from tensorflow.python.client import device_lib
from keras.backend.tensorflow_backend import set_session
import datetime

class PillRecognitionModel:
    def __init__(self, parameters):
        self.parameters = parameters

        self.workers = int(psutil.cpu_count()*0.8)

        self.trainFinalLayersFirst = (parameters['startingWeights']['weights'] == 'imagenet')
        self.pretrainEpochs = int(parameters['startingWeights'].get('pretrainPercent', 0) * parameters['neuralNetwork']['epochs'])
        if self.trainFinalLayersFirst:
            self.epochs = parameters['neuralNetwork']['epochs'] - self.pretrainEpochs
        else:
            self.epochs = parameters['neuralNetwork']['epochs']

        self.enableTensorboard = parameters['enableTensorboard']

        if parameters['numGPUs'] == -1:
            self.numGPUs = len(self.get_available_gpus())
        else:
            self.numGPUs = parameters['numGPUs']

        self.trainingBatchNumber = multiprocessing.Value('i', 0)
        self.trainingBatchNumberLock = multiprocessing.Lock()

        self.testingBatchNumber = multiprocessing.Value('i', 0)
        self.testingBatchNumberLock = multiprocessing.Lock()

        self.imageWidth = 224
        self.imageHeight = 224

        self.testingBatchSize = int(parameters['neuralNetwork']['batchSize'] * 2)
        self.datasetSizesToTest = parameters['datasetSizesToTest']
        self.rotationsToTest = parameters['rotationsToTest']
        self.testingPrintEvery = parameters['testingPrintEvery']
        self.measurementImages = []
        self.measurementRotatedImages = {}

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        self.augmentation = iaa.Sequential([
            iaa.Affine(
                scale=(parameters["augmentation"]["minScale"], parameters["augmentation"]["maxScale"]),
                translate_percent=(parameters["augmentation"]["minTranslate"], parameters["augmentation"]["maxTranslate"]),
                cval=255),
            iaa.PiecewiseAffine(parameters["augmentation"]["piecewiseAffine"]),
            iaa.GaussianBlur(sigma=(parameters["augmentation"]["minGaussianBlur"], parameters["augmentation"]["maxGaussianBlur"])),
            iaa.MotionBlur(k=(parameters["augmentation"]["minMotionBlur"], parameters["augmentation"]["maxMotionBlur"])),
            iaa.AdditiveGaussianNoise(scale=parameters["augmentation"]["gaussianNoise"] * 255)
        ])

        self.maxImagesToGenerate = 1000
        self.maxTripletsToGenerate = 1000
        self.imageGenerationManager = multiprocessing.Manager()
        self.imageGenerationExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=parameters['imageGenerationWorkers'])
        self.imageAugmentationExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=parameters['imageGenerationWorkers'])
        self.generatedImages = self.imageGenerationManager.list()
        self.augmentedTriplets = self.imageGenerationManager.list()
        self.imageGenerationThreads = []
        self.imageAugmentationThreads = []
        self.imageListLock = threading.Lock()
        self.augmentedTripletListLock = threading.Lock()
        self.running = False
        self.measuringAccuracy = False
        self.startTime = datetime.datetime.now()

        if parameters['augmentation']['rotationEasing']['easing'] == 'epoch':
            self.currentMaxRotation = self.imageGenerationManager.Value('i', 0)
        else:
            self.currentMaxRotation = self.imageGenerationManager.Value('i', parameters['augmentation']['maxRotation'])

        for n in range(parameters['imageGenerationWorkers']):
            newThread = threading.Thread(target=self.imageGenerationThread, daemon=False)
            newThread.start()
            self.imageGenerationThreads.append(newThread)

        time.sleep(2)

        for n in range(parameters['imageGenerationWorkers']-1):
            newThread = threading.Thread(target=self.imageAugmentationThread, daemon=False)
            newThread.start()
            self.imageAugmentationThreads.append(newThread)

    def imageGenerationThread(self):
        while threading.main_thread().isAlive():
            future = self.imageGenerationExecutor.submit(generatePillImage)
            image = future.result()
            with self.imageListLock:
                self.generatedImages.append(image)
                if len(self.generatedImages) >= self.maxImagesToGenerate:
                    del self.generatedImages[0]
            if not self.running or self.measuringAccuracy:
                time.sleep(0.6)

            if (datetime.datetime.now() - self.startTime).total_seconds() > 7000:
                exit(1)


    def imageAugmentationThread(self):
        while threading.main_thread().isAlive():
            future = self.imageAugmentationExecutor.submit(globalGenerateSingleTriplet, self.generatedImages, self.maxImagesToGenerate, self.currentMaxRotation, self.parameters, self.imageWidth, self.imageHeight, self.parameters['neuralNetwork']['vectorSize'])
            triplet = future.result()

            with self.augmentedTripletListLock:
                self.augmentedTriplets.append(triplet)
                if len(self.augmentedTriplets) >= self.maxTripletsToGenerate:
                    del self.augmentedTriplets[0]
            if not self.running or self.measuringAccuracy:
                time.sleep(0.3)

    def get_available_gpus(self):
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def create_triplet_loss(self):
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

                    anchorLosses = []
                    for d in range(self.parameters['neuralNetwork']['augmentationsPerImage']):
                        if d != k:
                            positive = tf.convert_to_tensor(y_pred[n*self.parameters['neuralNetwork']['augmentationsPerImage'] + d])

                            for j in range(int(self.parameters['neuralNetwork']['batchSize'])):
                                if j != n:
                                    for m in range(self.parameters['neuralNetwork']['augmentationsPerImage']):
                                        negative = tf.convert_to_tensor(y_pred[j * self.parameters['neuralNetwork']['augmentationsPerImage'] + m])

                                        pos_diff = tf.square(tf.subtract(anchor, positive))
                                        neg_diff = tf.square(tf.subtract(anchor, negative))

                                        alpha = 0.2

                                        loss = None
                                        if self.parameters['lossFunction']['transformSumMode'] == 'summed':
                                            # distance between the anchor and the positive
                                            pos_dist = tf.reduce_sum(pos_diff)
                                            # distance between the anchor and the negative
                                            neg_dist = tf.reduce_sum(neg_diff)

                                            beta = self.parameters['neuralNetwork']['vectorSize'] + 1

                                            if self.parameters['lossFunction']['transform'] == "linear":
                                                pos_dist = tf.divide((pos_dist), beta)
                                                neg_dist = tf.divide((self.parameters['neuralNetwork']['vectorSize'] - neg_dist), beta)
                                                loss = neg_dist * self.parameters['lossFunction']['negativeWeight'] + pos_dist * self.parameters['lossFunction']['positiveWeight']
                                            elif self.parameters['lossFunction']['transform'] == "max":
                                                pos_dist = tf.divide((pos_dist), beta)
                                                neg_dist = tf.divide((neg_dist), beta)
                                                loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
                                            elif self.parameters['lossFunction']['transform'] == "logarithmic":
                                                pos_dist = -tf.log(-tf.divide((pos_dist), beta) + 1 + epsilon)
                                                neg_dist = -tf.log(-tf.divide((self.parameters['neuralNetwork']['vectorSize'] - neg_dist), beta) + 1 + epsilon)
                                                loss = neg_dist * self.parameters['lossFunction']['negativeWeight'] + pos_dist * self.parameters['lossFunction']['positiveWeight']
                                        else:
                                            if self.parameters['lossFunction']['transform'] == "linear":
                                                pos_transformed = pos_diff
                                                neg_transformed = 1.0 - neg_diff
                                                pos_loss = tf.reduce_mean(pos_transformed, axis=1)
                                                neg_loss = tf.reduce_mean(neg_transformed, axis=1)
                                                loss = neg_loss * self.parameters['lossFunction']['negativeWeight'] + pos_loss * self.parameters['lossFunction']['positiveWeight']

                                            elif self.parameters['lossFunction']['transform'] == "max":
                                                # -ln(-x/N+1)
                                                loss = tf.reduce_mean(tf.maximum(pos_diff - neg_diff + alpha, 0), axis=1)
                                            elif self.parameters['lossFunction']['transform'] == "logarithmic":
                                                # -ln(-x/N+1)
                                                pos_transformed = -tf.log(-pos_diff + 1 + epsilon)
                                                neg_transformed = -tf.log(-(1.0 - neg_diff) + 1 + epsilon)
                                                pos_loss = tf.reduce_mean(pos_transformed, axis=1)
                                                neg_loss = tf.reduce_mean(neg_transformed, axis=1)
                                                loss = neg_loss * self.parameters['lossFunction']['negativeWeight'] + pos_loss * self.parameters['lossFunction']['positiveWeight']

                                        anchorLosses.append(loss)
                    if self.parameters['lossFunction']['batchMode'] == 'hard':
                        anchorLoss = tf.reduce_max(tf.stack(anchorLosses))
                    elif self.parameters['lossFunction']['batchMode'] == 'all':
                        anchorLoss = tf.reduce_mean(tf.stack(anchorLosses))
                    losses.append(anchorLoss)
            return tf.reduce_mean(tf.stack(anchorLosses))
        return triplet_loss

    def generateBatch(self, testing=False):
        while True:
            # Generate half the batch as negative examples, half the batch as positive examples
            inputs = []
            outputs = []

            triplets = random.sample(range(min(len(self.augmentedTriplets)-1, self.maxTripletsToGenerate-1)), int(self.parameters['neuralNetwork']['batchSize']))

            # Augment the images
            for n in range(int(self.parameters['neuralNetwork']['batchSize'])):
                tripletInputs, tripletOutputs = self.augmentedTriplets[triplets[n]]

                for input in range(self.parameters['neuralNetwork']['augmentationsPerImage']):
                    inputs.append(tripletInputs[input])
                    outputs.append(tripletOutputs)

            yield numpy.array(inputs), numpy.array(outputs)

    def trainModel(self):
        primaryDevice = "/cpu:0"
        if self.numGPUs == 1:
            primaryDevice = "/gpu:0"

        with tf.device(primaryDevice):
            input = Input((self.imageWidth, self.imageHeight, 3))

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
            imageNet.add(Reshape([-1]))
            imageNet.add(BatchNormalization())
            imageNet.add(Dropout(self.parameters["neuralNetwork"]["dropoutRate"]))
            imageNet.add(Dense(int(self.parameters['neuralNetwork']['vectorSize']*self.parameters["neuralNetwork"]["denseLayerMultiplier"]), activation=self.parameters["neuralNetwork"]["denseActivation"]))
            imageNet.add(BatchNormalization())
            imageNet.add(Dense(int(self.parameters['neuralNetwork']['vectorSize']), activation=self.parameters["neuralNetwork"]["finalActivation"]))

            imageNet.summary()

            encoded_predict = imageNet(input)

            # Create one model for training using triplet loss, and another model for live prediction
            model = Model(inputs=[input], outputs=encoded_predict)

        if self.numGPUs > 1:
            model = multi_gpu_model(model, gpus=self.numGPUs)


        testingGenerator = self.generateBatch(testing=True)
        trainingGenerator = self.generateBatch(testing=False)

        bestAccuracy = None

        def epochCallback(epoch, logs):
            nonlocal bestAccuracy
            self.measuringAccuracy=True
            if self.parameters['augmentation']['rotationEasing']['easing'] == 'epoch':
                self.currentMaxRotation.value = min(1.0, float(epoch) / float((self.parameters['augmentation']['rotationEasing']['rotationEasing'] * self.parameters['neuralNetwork']['epochs']))) * self.parameters['augmentation']['maxRotation']

            if epoch % self.parameters['epochsBeforeAccuracyMeasurement'] == (self.parameters['epochsBeforeAccuracyMeasurement']-1):
                accuracy = self.measureAccuracy(model)
                if bestAccuracy is None or accuracy > bestAccuracy:
                    bestAccuracy = accuracy
            self.measuringAccuracy = False

        testNearestNeighbor = LambdaCallback(on_epoch_end=epochCallback)

        reduceLRCallback = ReduceLROnPlateau(monitor='loss', factor=self.parameters["neuralNetwork"]["reduceLRFactor"], patience=self.parameters["neuralNetwork"]["reduceLRPatience"], verbose=1)

        callbacks = [testNearestNeighbor, reduceLRCallback]
        optimizer = None


        if self.enableTensorboard:
            tensorBoardCallback = TensorBoard(
                log_dir='./logs',
                histogram_freq=0,
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
                optimizer = Adam(self.parameters["neuralNetwork"]["optimizer"]["pretrainLearningRate"])
            elif self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'nadam':
                optimizer = Nadam(self.parameters["neuralNetwork"]["optimizer"]["pretrainLearningRate"])
            elif self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'rmsprop':
                optimizer = RMSprop(self.parameters["neuralNetwork"]["optimizer"]["pretrainLearningRate"])
            elif self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'sgd':
                optimizer = SGD(self.parameters["neuralNetwork"]["optimizer"]["pretrainLearningRate"])

            model.compile(loss=self.create_triplet_loss(), optimizer=optimizer)

            model.summary()
            model.count_params()
            self.running = True

            model.fit_generator(
                generator=trainingGenerator,
                steps_per_epoch=self.parameters['stepsPerEpoch'],
                epochs=self.pretrainEpochs,
                validation_data=testingGenerator,
                validation_steps=self.parameters['validationSteps'],
                workers=1,
                use_multiprocessing=False,
                max_queue_size=self.parameters['maxQueueSize'],
                callbacks=callbacks
            )

        imageNet.layers[0].trainable = True

        if self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'adam':
            optimizer = Adam(self.parameters["neuralNetwork"]["optimizer"]["learningRate"])
        elif self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'nadam':
            optimizer = Nadam(self.parameters["neuralNetwork"]["optimizer"]["learningRate"])
        elif self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'rmsprop':
            optimizer = RMSprop(self.parameters["neuralNetwork"]["optimizer"]["learningRate"])
        elif self.parameters['neuralNetwork']['optimizer']['optimizerName'] == 'sgd':
            optimizer = SGD(self.parameters["neuralNetwork"]["optimizer"]["learningRate"])


        model.compile(loss=self.create_triplet_loss(), optimizer=optimizer)

        model.summary()
        model.count_params()
        self.running = True

        model.fit_generator(
            generator=trainingGenerator,
            steps_per_epoch=self.parameters['stepsPerEpoch'],
            epochs=self.epochs,
            validation_data=testingGenerator,
            validation_steps=self.parameters['validationSteps'],
            workers=1,
            use_multiprocessing=False,
            max_queue_size=self.parameters['maxQueueSize'],
            callbacks=callbacks
        )

        return bestAccuracy

    def measureAccuracy(self, model):
        print("Measuring Accuracy", flush=True)
        maxDatasetSize = max(*self.datasetSizesToTest)

        if len(self.measurementImages) < maxDatasetSize:
            self.measurementRotatedImages = {
                rotation: [] for rotation in self.rotationsToTest
            }

            print("    Generating test images", flush=True)
            completedImages = 0
            # Build images in sets of 10 batches. This is to get around a python multiprocessing bug.
            for k in range(10):
                futures = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as worker:
                    for n in range(int(maxDatasetSize/10)):
                        futures.append(worker.submit(globalGenerateTestImages, self.imageWidth, self.imageHeight, self.rotationsToTest))

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
        allAccuracies = []
        accuracyRows = []
        for rotation in self.rotationsToTest:
            if len(originalVectors[rotation]) == 0:
                print(f"        Error! Unable to measure accuracy for rotation {rotation}. All samples had bad vectors.", flush=True)
                continue

            sizeAccuracies = {
                "rot": rotation
            }

            for datasetSize in self.datasetSizesToTest:
                print(f"        Measuring accuracy on {rotation} degree rotations with dataset size {datasetSize}", flush=True)

                vectorIndexes = random.sample(range(len(originalVectors[rotation])), min(len(originalVectors[rotation]), datasetSize))

                origVectorsForTest = [originalVectors[rotation][index] for index in vectorIndexes]
                rotatedVectorsForTest = [rotatedVectors[rotation][index] for index in vectorIndexes]

                print(f"            Fitting the nearest neighbor model on {len(origVectorsForTest)} samples", flush=True)
                nearestNeighborModel = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
                nearestNeighborModel.fit(origVectorsForTest)

                print(f"            Computing results of nearest neighbors model on rotated images.", flush=True)
                distance, indices = nearestNeighborModel.kneighbors(rotatedVectorsForTest)

                correct = 0
                for n in range(len(origVectorsForTest)):
                    if indices[n] == n:
                        correct += 1

                accuracy = float(correct) / float(len(origVectorsForTest))
                print(f"            Nearest Neighbor Accuracy on {rotation} degree rotations with {datasetSize} total dataset size: {accuracy}", flush=True)
                allAccuracies.append(accuracy)
                sizeAccuracies[f's_{str(datasetSize)}'] = accuracy
            accuracyRows.append(sizeAccuracies)

        if len(allAccuracies) == 0:
            return 0

        writer = csv.DictWriter(sys.stdout, fieldnames=list(accuracyRows[0].keys()), dialect=csv.excel_tab)
        writer.writeheader()
        writer.writerows(accuracyRows)
        print("")
        meanAccuracy = numpy.mean(allAccuracies)
        print(f"    Final Mean Accuracy {meanAccuracy}", flush=True)
        return meanAccuracy


def globalGenerateTestImages(width, height, rotations):
    image = generatePillImage()
    image = skimage.transform.resize(image, (width, height, 3), mode='reflect', anti_aliasing=True)
    rotated = {rotation: skimage.transform.rotate(image, angle=rotation, mode='constant', cval=1) for rotation in rotations}
    return image, rotated

def globalGenerateSingleTriplet(generatedImages, maxImagesToGenerate, currentMaxRotation, parameters, imageWidth, imageHeight, vectorSize):
    augmentation =  iaa.Sequential([
            iaa.Affine(
                scale=(parameters["augmentation"]["minScale"], parameters["augmentation"]["maxScale"]),
                translate_percent=(parameters["augmentation"]["minTranslate"], parameters["augmentation"]["maxTranslate"]),
                cval=255),
            iaa.PiecewiseAffine(parameters["augmentation"]["piecewiseAffine"]),
            iaa.GaussianBlur(sigma=(parameters["augmentation"]["minGaussianBlur"], parameters["augmentation"]["maxGaussianBlur"])),
            iaa.MotionBlur(k=(parameters["augmentation"]["minMotionBlur"], parameters["augmentation"]["maxMotionBlur"])),
            iaa.AdditiveGaussianNoise(scale=parameters["augmentation"]["gaussianNoise"] * 255)
        ])

    anchor = random.randint(0, min(len(generatedImages)-1, maxImagesToGenerate-1))
    anchor = generatedImages[anchor]

    augmentations = []    
    for n in range(parameters['neuralNetwork']['augmentationsPerImage']):
        anchorAugmented = skimage.transform.rotate(anchor, angle=random.uniform(-currentMaxRotation.value / 2, currentMaxRotation.value / 2), mode='constant', cval=1)
        anchorAugmented = augmentation.augment_images(numpy.array([anchorAugmented]) * 255.0)[0]
        anchorAugmented /= 255.0
        anchorAugmented = skimage.transform.resize(anchorAugmented, (imageWidth, imageHeight, 3), mode='reflect', anti_aliasing=True)
        augmentations.append(anchorAugmented)

    return (numpy.array(augmentations), numpy.ones(int(vectorSize)) * 1.0)
