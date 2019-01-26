from keras.applications import MobileNetV2, InceptionV3, ResNet50, NASNetMobile, NASNetLarge
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Reshape, Input, Flatten, Subtract, Lambda, Concatenate
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
            self.numGPUs = parameters['numGPUs']

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

        self.maxImagesToGenerate = 100
        self.imageGenerationManager = multiprocessing.Manager()
        self.imageGenerationExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=parameters['imageGenerationWorkers'])
        self.augmentedImages = self.imageGenerationManager.list()
        self.imageGenerationThreads = []
        self.imageListLock = threading.Lock()
        self.augmentedTripletListLock = threading.Lock()
        self.running = False
        self.measuringAccuracy = False
        self.startTime = datetime.datetime.now()

        if parameters['augmentation']['rotationEasing']['easing'] == 'epoch':
            self.currentMinRotation = self.imageGenerationManager.Value('i', 0)
            self.currentMaxRotation = self.imageGenerationManager.Value('i', 1)
        else:
            self.currentMinRotation = self.imageGenerationManager.Value('i', parameters['augmentation']['minRotation'])
            self.currentMaxRotation = self.imageGenerationManager.Value('i', parameters['augmentation']['maxRotation'])

        time.sleep(2)

        for n in range(parameters['imageGenerationWorkers']):
            newThread = threading.Thread(target=self.imageGenerationThread, daemon=False)
            newThread.start()
            self.imageGenerationThreads.append(newThread)

    def imageGenerationThread(self):
        while threading.main_thread().isAlive():
            future = self.imageGenerationExecutor.submit(globalGenerateSingleTriplet, self.currentMaxRotation, self.currentMinRotation, self.parameters, self.imageWidth, self.imageHeight, self.parameters['neuralNetwork']['vectorSize'])
            triplet = future.result()

            with self.augmentedTripletListLock:
                self.augmentedImages.append(triplet)
                if len(self.augmentedImages) >= self.maxImagesToGenerate:
                    del self.augmentedImages[0]
            if not self.running or self.measuringAccuracy:
                time.sleep(1.0)

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

                    worstPosDist = tf.reduce_max(tf.stack(posDists))
                    worstNegDist = tf.reduce_min(tf.stack(negDists))

                    margin = 0.3

                    pos_dist = tf.divide((worstPosDist), self.parameters['neuralNetwork']['vectorSize'])
                    neg_dist = tf.divide((worstNegDist), self.parameters['neuralNetwork']['vectorSize'])
                    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)

                    losses.append(loss)

            losses = tf.stack(losses)
            nonZeroEntries = tf.boolean_mask(losses, tf.not_equal(losses, tf.zeros_like(losses)))  # [0, 2]

            is_empty = tf.equal(tf.size(nonZeroEntries), 0)

            loss = tf.cond( is_empty,
                            lambda: tf.constant(0, tf.float32),
                            lambda: tf.reduce_mean(losses))

            return  loss
        return triplet_loss

    def generateBatch(self, testing=False):
        while True:
            # Generate half the batch as negative examples, half the batch as positive examples
            inputs = []
            outputs = []

            triplets = random.sample(range(min(len(self.augmentedImages)-1, self.maxImagesToGenerate-1)), int(self.parameters['neuralNetwork']['batchSize']))

            # Augment the images
            for n in range(int(self.parameters['neuralNetwork']['batchSize'])):
                tripletInputs, tripletOutputs = self.augmentedImages[triplets[n]]

                for input in range(self.parameters['neuralNetwork']['augmentationsPerImage']):
                    inputs.append(tripletInputs[input])
                    outputs.append(tripletOutputs)

            yield numpy.array(inputs), numpy.array(outputs)

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
            imageNet.add(Dense(int(self.parameters['neuralNetwork']['vectorSize']*self.parameters["neuralNetwork"]["denseLayerMultiplier"]), activation=self.parameters["neuralNetwork"]["denseActivation"]))
            imageNet.add(BatchNormalization())
            imageNet.add(Dense(int(self.parameters['neuralNetwork']['vectorSize']), activation=self.parameters["neuralNetwork"]["finalActivation"]))

            imageNet.summary()

        if self.numGPUs > 1:
            model = multi_gpu_model(imageNet, gpus=self.numGPUs)
        else:
            model = imageNet
        
        return model, imageNet

    def trainModel(self):
        self.model, imageNet = self.createCoreModel()

        testingGenerator = self.generateBatch(testing=True)
        trainingGenerator = self.generateBatch(testing=False)

        bestAccuracy = None

        def epochCallback(epoch, logs):
            nonlocal bestAccuracy
            self.measuringAccuracy=True
            if self.parameters['augmentation']['rotationEasing']['easing'] == 'epoch':
                self.currentMinRotation.value = min(1.0, float(epoch) / float((self.parameters['augmentation']['rotationEasing']['minRotationEasing'] * self.parameters['neuralNetwork']['epochs']))) * self.parameters['augmentation']['minRotation']
                self.currentMaxRotation.value = min(1.0, float(epoch) / float((self.parameters['augmentation']['rotationEasing']['maxRotationEasing'] * self.parameters['neuralNetwork']['epochs']))) * self.parameters['augmentation']['maxRotation']

            if epoch % 5 == 0:
                imageNet.save(f"model-epoch-{epoch}.h5")
                imageNet.save_weights(f"model-epoch-{epoch}-weights.h5")
            imageNet.save(f"model-current.h5")
            imageNet.save_weights(f"model-current-weights.h5")

            if epoch % self.parameters['epochsBeforeAccuracyMeasurement'] == (self.parameters['epochsBeforeAccuracyMeasurement']-1):
                accuracy = self.measureAccuracy(self.model)
                if bestAccuracy is None or accuracy > bestAccuracy:
                    bestAccuracy = accuracy
            self.measuringAccuracy = False

        rollingAverage9 = None
        rollingAverage95 = None
        rollingAverage99 = None
        rollingAverage995 = None
        def batchCallback(batch, log):
            nonlocal rollingAverage9, rollingAverage95, rollingAverage99, rollingAverage995
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

        reduceLRCallback = ReduceLROnPlateau(monitor='loss', factor=self.parameters["neuralNetwork"]["reduceLRFactor"], patience=self.parameters["neuralNetwork"]["reduceLRPatience"], verbose=1)

        callbacks = [testNearestNeighbor, reduceLRCallback]
        optimizer = None

        imageNet.load_weights('model-current-weights.h5')

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


        self.model.compile(loss=self.createTripletLoss(), optimizer=optimizer)

        self.model.summary()
        self.model.count_params()
        self.running = True

        self.model.fit_generator(
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

        imageNet.save(f"model-final.h5")
        imageNet.save_weights(f"model-final-weights.h5")

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


def globalGenerateTestImages(width, height, rotations):
    image = generatePillImage()
    image = skimage.transform.resize(image, (width, height, 3), mode='reflect', anti_aliasing=True)
    rotated = {rotation: skimage.transform.rotate(image, angle=rotation, mode='constant', cval=1) for rotation in rotations}
    return image, rotated

def globalGenerateSingleTriplet(currentMaxRotation, currentMinRotation, parameters, imageWidth, imageHeight, vectorSize):
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

    anchor = generatePillImage()

    augmentations = []    
    for n in range(parameters['neuralNetwork']['augmentationsPerImage']):
        rotationDirection = random.choice([-1, +1])
        anchorAugmented = skimage.transform.rotate(anchor, angle=random.uniform(currentMinRotation.value/2, currentMaxRotation.value/2) * rotationDirection, mode='constant', cval=1)
        anchorAugmented = augmentation.augment_images(numpy.array([anchorAugmented]) * 255.0)[0]
        anchorAugmented /= 255.0
        anchorAugmented = skimage.transform.resize(anchorAugmented, (imageWidth, imageHeight, 3), mode='reflect', anti_aliasing=True)
        augmentations.append(anchorAugmented)

    return (numpy.array(augmentations), numpy.ones(int(vectorSize)) * 1.0)
