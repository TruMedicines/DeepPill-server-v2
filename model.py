from keras.applications import MobileNetV2, InceptionV3
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Reshape, Input, merge, Flatten, Subtract, Lambda, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, TensorBoard
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
from pprint import pprint
from generate_data import generatePillImage
import math
import skimage.transform
import random
import concurrent.futures
import multiprocessing
from tensorflow.python.client import device_lib


class PillRecognitionModel:
    def __init__(self, parameters):
        self.parameters = parameters

        self.batchSize = parameters['batchSize']
        self.minRotation = parameters['minRotation']
        self.maxRotation = parameters['maxRotation']
        self.vectorSize = parameters['vectorSize']
        self.workers = min(16, int(psutil.cpu_count()*0.9))
        self.stepsPerEpoch = parameters['stepsPerEpoch']
        self.firstPassEpochs = parameters['firstPassEpochs']
        self.secondPassEpochs = parameters['secondPassEpochs']
        self.validationSteps = parameters['validationSteps']
        self.maxQueueSize = parameters['maxQueueSize']
        self.epochsBeforeAccuracyMeasurement = parameters['epochsBeforeAccuracyMeasurement']
        self.firstPassLearningRate = parameters['firstPassLearningRate']
        self.secondPassLearningRate = parameters['secondPassLearningRate']
        self.denseDropoutRate = parameters['denseDropoutRate']
        self.denseFirstLayerSizeMultiplier = parameters['denseFirstLayerSizeMultiplier']
        self.denseActivation = parameters['denseActivation']
        self.finalActivation = parameters['finalActivation']
        self.startingWeights = parameters['startingWeights']

        self.minScale = parameters["minScale"]
        self.maxScale = parameters["maxScale"]
        self.minTranslate = parameters["minTranslate"]
        self.maxTranslate = parameters["maxTranslate"]
        self.piecewiseAffine = parameters["piecewiseAffine"]
        self.minGaussianBlur = parameters["minGaussianBlur"]
        self.maxGaussianBlur = parameters["maxGaussianBlur"]
        self.minMotionBlur = parameters["minMotionBlur"]
        self.maxMotionBlur = parameters["maxMotionBlur"]
        self.gaussianNoiseStrength = parameters["gaussianNoiseStrength"]

        self.enableTensorboard = parameters['enableTensorboard']

        if parameters['numGPUs'] == -1:
            self.numGPUs = len(self.get_available_gpus())
        else:
            self.numGPUs = parameters['numGPUs']

        self.trainingBatchNumber = multiprocessing.Value('i', 0)
        self.trainingBatchNumberLock = multiprocessing.Lock()

        self.testingBatchNumber = multiprocessing.Value('i', 0)
        self.testingBatchNumberLock = multiprocessing.Lock()

        self.testingBatchSize = parameters['testingBatchSize']
        self.datasetSizesToTest = parameters['datasetSizesToTest']
        self.rotationsToTest = parameters['rotationsToTest']
        self.testingPrintEvery = parameters['testingPrintEvery']
        self.measurementImages = []
        self.measurementRotatedImages = {}

        self.augmentation = iaa.Sequential([
            iaa.Affine(
                scale=(self.minScale, self.maxScale),
                translate_percent=(self.minTranslate, self.maxTranslate),
                rotate=(self.minRotation, self.maxRotation),
                cval=255),
            iaa.PiecewiseAffine(self.piecewiseAffine),
            iaa.GaussianBlur(sigma=(self.minGaussianBlur, self.maxGaussianBlur)),
            iaa.MotionBlur(k=(self.minMotionBlur, self.maxMotionBlur)),
            iaa.AdditiveGaussianNoise(scale=self.gaussianNoiseStrength * 255)
        ])

    def get_available_gpus(self):
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def create_triplet_loss(self):
        def triplet_loss(y_true, y_pred, epsilon=1e-8):
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
            anchor = tf.convert_to_tensor(y_pred[:, 0:self.vectorSize])
            positive = tf.convert_to_tensor(y_pred[:, self.vectorSize:self.vectorSize*2])
            negative = tf.convert_to_tensor(y_pred[:, self.vectorSize*2:self.vectorSize*3])

            # distance between the anchor and the positive
            pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
            # distance between the anchor and the negative
            neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

            # -ln(-x/N+1)
            pos_dist = -tf.log(-tf.divide((pos_dist), self.vectorSize) + 1 + epsilon)
            neg_dist = -tf.log(-tf.divide((self.vectorSize - neg_dist), self.vectorSize) + 1 + epsilon)

            # compute loss
            loss = neg_dist + pos_dist

            return loss
        return triplet_loss

    def generateBatch(self, testing=False):
        while True:
            batchNumber = 0
            if not testing:
                with self.trainingBatchNumberLock:
                    self.trainingBatchNumber.value += 1
                    batchNumber = self.trainingBatchNumber.value
            else:
                with self.testingBatchNumberLock:
                    self.testingBatchNumber.value += 1
                    batchNumber = self.testingBatchNumber.value

            # print(batchNumber, flush=True)

            # Generate half the batch as negative examples, half the batch as positive examples
            inputs = []
            outputs = []

            # Generate negatives as just separate images
            for n in range(int(self.batchSize)):
                anchor = generatePillImage()
                negative = generatePillImage()

                anchorAugmented, positiveAugmented, negativeAugmented = self.augmentation.augment_images(numpy.array([anchor, copy.copy(anchor), negative]) * 255.0)
                anchorAugmented /= 255.0
                positiveAugmented /= 255.0
                negativeAugmented /= 255.0

                # anchorAugmented = skimage.transform.rotate(anchor, angle=random.choice([1, -1]) * random.uniform(0, self.minRotation), mode='constant', cval=1)
                # positiveAugmented = skimage.transform.rotate(anchor, angle=random.choice([1, -1]) * random.uniform(self.minRotation, self.maxRotation), mode='constant', cval=1)
                # negativeAugmented = skimage.transform.rotate(negative, angle=random.choice([1, -1]) * random.uniform(self.minRotation, self.maxRotation), mode='constant', cval=1)

                inputs.append(numpy.array([anchorAugmented, positiveAugmented, negativeAugmented]))
                outputs.append(numpy.ones(self.vectorSize) * 1.0)

                # plt.imshow(anchorAugmented, interpolation='lanczos')
                # plt.savefig(f'example-{batchNumber}-{n}-anchor.png')
                #
                # plt.imshow(positiveAugmented, interpolation='lanczos')
                # plt.savefig(f'example-{batchNumber}-{n}-positive.png')
                #
                # plt.imshow(negativeAugmented, interpolation='lanczos')
                # plt.savefig(f'example-{batchNumber}-{n}-negative.png')

            yield numpy.array(inputs), numpy.array(outputs)

    def trainModel(self):
        primaryDevice = "/cpu:0"
        if self.numGPUs == 1:
            primaryDevice = "/gpu:0"

        with tf.device(primaryDevice):
            trainingPrimary = Input((3, 256, 256, 3))
            predictPrimary = Input((256, 256, 3))

            imageNet = Sequential()
            imageNetCore = InceptionV3(include_top=False, pooling=None, input_shape=(256, 256, 3), weights=self.parameters['startingWeights'])

            imageNet.add(imageNetCore)
            imageNet.add(Reshape([-1]))
            imageNet.add(BatchNormalization())
            imageNet.add(Dropout(self.denseDropoutRate))
            imageNet.add(Dense(int(self.vectorSize*self.denseFirstLayerSizeMultiplier), activation=self.denseActivation))
            imageNet.add(BatchNormalization())
            imageNet.add(Dropout(self.denseDropoutRate))
            imageNet.add(Dense(self.vectorSize, activation=self.finalActivation))

            encoded_anchor = imageNet(Lambda(lambda x: x[:, 0])(trainingPrimary))
            encoded_positive = imageNet(Lambda(lambda x: x[:, 1])(trainingPrimary))
            encoded_negative = imageNet(Lambda(lambda x: x[:, 2])(trainingPrimary))

            encoded_triplet = Concatenate(axis=1)([encoded_anchor, encoded_positive, encoded_negative])

            encoded_predict = imageNet(predictPrimary)

            # Create one model for training using triplet loss, and another model for live prediction
            trainingModel = Model(inputs=[trainingPrimary], outputs=encoded_triplet)
            predictionModel = Model(inputs=[predictPrimary], outputs=encoded_predict)

        if self.numGPUs > 1:
            trainingModel = multi_gpu_model(trainingModel, gpus=self.numGPUs)
            predictionModel = multi_gpu_model(predictionModel, gpus=self.numGPUs)

        imageNet.layers[0].trainable = False

        optimizer = Adam(self.firstPassLearningRate)
        trainingModel.compile(loss=self.create_triplet_loss(), optimizer=optimizer)

        trainingModel.summary()
        trainingModel.count_params()
        predictionModel.summary()
        predictionModel.count_params()

        testingGenerator = self.generateBatch(testing=True)
        trainingGenerator = self.generateBatch(testing=False)

        def epochCallback(epoch, logs):
            predictionModel.compile(loss="mean_squared_error", optimizer=optimizer)
            if epoch % self.epochsBeforeAccuracyMeasurement == (self.epochsBeforeAccuracyMeasurement-1):
                self.measureAccuracy(predictionModel)

        testNearestNeighbor = LambdaCallback(on_epoch_end=epochCallback)

        callbacks = [testNearestNeighbor]

        if self.enableTensorboard:
            tensorBoardCallback = TensorBoard(
                log_dir='./logs',
                histogram_freq=0,
                batch_size=self.batchSize,
                write_graph=True,
                write_grads=False,
                write_images=False,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None,
                embeddings_data=None,
                update_freq='batch')
            callbacks.append(tensorBoardCallback)

        trainingModel.fit_generator(
            generator=trainingGenerator,
            steps_per_epoch=self.stepsPerEpoch,
            epochs=self.firstPassEpochs,
            validation_data=testingGenerator,
            validation_steps=self.validationSteps,
            workers=self.workers,
            use_multiprocessing=True,
            max_queue_size=self.maxQueueSize,
            callbacks=callbacks
        )

        # K.clear_session()

        imageNet.layers[0].trainable = True

        optimizer = Adam(self.secondPassLearningRate)
        trainingModel.compile(loss=self.create_triplet_loss(), optimizer=optimizer)

        trainingModel.summary()
        trainingModel.count_params()

        trainingModel.fit_generator(
            generator=trainingGenerator,
            steps_per_epoch=self.stepsPerEpoch,
            epochs=self.secondPassEpochs,
            validation_data=testingGenerator,
            validation_steps=self.validationSteps,
            workers=self.workers,
            use_multiprocessing=True,
            max_queue_size=self.maxQueueSize,
            callbacks=callbacks
        )

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
                with concurrent.futures.ProcessPoolExecutor(max_workers=int(psutil.cpu_count() * 0.9)) as worker:
                    for n in range(int(maxDatasetSize/10)):
                        futures.append(worker.submit(globalGenerateTestImages, self.rotationsToTest))

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
                "rotation": rotation
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
                sizeAccuracies[f'size_{str(datasetSize)}'] = accuracy
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


def globalGenerateTestImages(rotations):
    image = generatePillImage()
    rotated = {rotation: skimage.transform.rotate(image, angle=rotation, mode='constant', cval=1) for rotation in rotations}
    return image, rotated
