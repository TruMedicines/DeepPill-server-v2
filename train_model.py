from keras.applications import MobileNetV2, InceptionV3
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Reshape, Input, merge, Flatten, Subtract, Lambda, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, TensorBoard
from keras.utils import multi_gpu_model
from keras.regularizers import l1, l2
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
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
import sklearn.neighbors
import json
from pprint import pprint
from generate_data import generatePillImage
import math
import skimage.transform
import random
import concurrent.futures
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def create_triplet_loss(vectorSize):
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
        anchor = tf.convert_to_tensor(y_pred[:, 0:vectorSize])
        positive = tf.convert_to_tensor(y_pred[:, vectorSize:vectorSize*2])
        negative = tf.convert_to_tensor(y_pred[:, vectorSize*2:vectorSize*3])

        # distance between the anchor and the positive
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        # distance between the anchor and the negative
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        # -ln(-x/N+1)
        pos_dist = -tf.log(-tf.divide((pos_dist), vectorSize) + 1 + epsilon)
        neg_dist = -tf.log(-tf.divide((vectorSize - neg_dist), vectorSize) + 1 + epsilon)

        # compute loss
        loss = neg_dist + pos_dist

        return loss
    return triplet_loss


def trainModel():
    batchSize = 32
    batchNumber = 0
    minRotation = 5
    maxRotation = 30
    vectorSize = 1024
    workers = min(16, int(psutil.cpu_count()*0.9))
    stepsPerEpoch = 100
    firstPassEpochs = 20
    secondPassEpochs = 5000
    validationSteps = 10
    maxQueueSize = 10
    epochsBeforeAccuracyMeasurement = 2
    firstPassLearningRate = 1e-3
    secondPassLearningRate = 1e-4
    denseDropoutRate = 0.5
    denseFirstLayerSizeMultiplier = 2
    denseActivation = 'softsign'
    finalActivation = 'sigmoid'
    numGPUs = len(get_available_gpus())
    # numGPUs = 1

    def generateBatch():
        nonlocal batchNumber
        while True:
            batchNumber += 1

            # Generate half the batch as negative examples, half the batch as positive examples
            inputs = []
            outputs = []

            # Generate negatives as just separate images
            for n in range(int(batchSize)):
                anchor = generatePillImage()
                negative = generatePillImage()

                anchorRotated = skimage.transform.rotate(anchor, angle=random.uniform(0, minRotation), mode='constant', cval=1)
                positiveRotated = skimage.transform.rotate(anchor, angle=random.uniform(minRotation, maxRotation), mode='constant', cval=1)

                negativeRotated = skimage.transform.rotate(negative, angle=random.uniform(minRotation, maxRotation), mode='constant', cval=1)

                inputs.append(numpy.array([anchorRotated, positiveRotated, negativeRotated]))
                outputs.append(numpy.ones(vectorSize)*1.0)

            yield numpy.array(inputs), numpy.array(outputs)

    primaryDevice = "/cpu:0"
    if numGPUs == 1:
        primaryDevice = "/gpu:0"

    with tf.device(primaryDevice):
        trainingPrimary = Input((3, 256, 256, 3))
        predictPrimary = Input((256, 256, 3))

        imageNet = Sequential()
        imageNetCore = InceptionV3(include_top=False, pooling=None, input_shape=(256, 256, 3), weights='imagenet')

        imageNet.add(imageNetCore)
        imageNet.add(Reshape([-1]))
        imageNet.add(BatchNormalization())
        imageNet.add(Dropout(denseDropoutRate))
        imageNet.add(Dense(int(vectorSize*denseFirstLayerSizeMultiplier), activation=denseActivation))
        imageNet.add(BatchNormalization())
        imageNet.add(Dropout(denseDropoutRate))
        imageNet.add(Dense(vectorSize, activation=finalActivation))

        encoded_anchor = imageNet(Lambda(lambda x: x[:, 0])(trainingPrimary))
        encoded_positive = imageNet(Lambda(lambda x: x[:, 1])(trainingPrimary))
        encoded_negative = imageNet(Lambda(lambda x: x[:, 2])(trainingPrimary))

        encoded_triplet = Concatenate(axis=1)([encoded_anchor, encoded_positive, encoded_negative])

        encoded_predict = imageNet(predictPrimary)

        # Create one model for training using triplet loss, and another model for live prediction
        trainingModel = Model(inputs=[trainingPrimary], outputs=encoded_triplet)
        predictionModel = Model(inputs=[predictPrimary], outputs=encoded_predict)

    if numGPUs > 1:
        trainingModel = multi_gpu_model(trainingModel, gpus=numGPUs)
        predictionModel = multi_gpu_model(predictionModel, gpus=numGPUs)

    imageNet.layers[0].trainable = False

    optimizer = Adam(firstPassLearningRate)
    trainingModel.compile(loss=create_triplet_loss(vectorSize), optimizer=optimizer)

    trainingModel.summary()
    trainingModel.count_params()
    predictionModel.summary()
    predictionModel.count_params()

    testingGenerator = generateBatch()
    trainingGenerator = generateBatch()

    def epochCallback(epoch, logs):
        predictionModel.compile(loss="mean_squared_error", optimizer=optimizer)
        if epoch % epochsBeforeAccuracyMeasurement == (epochsBeforeAccuracyMeasurement-1):
            measureAccuracy(predictionModel)

    testNearestNeighbor = LambdaCallback(on_epoch_end=epochCallback)

    tensorBoardCallback = TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        batch_size=batchSize,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq='batch')

    trainingModel.fit_generator(
        generator=trainingGenerator,
        steps_per_epoch=stepsPerEpoch,
        epochs=firstPassEpochs,
        validation_data=testingGenerator,
        validation_steps=validationSteps,
        workers=workers,
        use_multiprocessing=True,
        max_queue_size=maxQueueSize,
        callbacks=[testNearestNeighbor, tensorBoardCallback]
    )

    # K.clear_session()

    imageNet.layers[0].trainable = True

    optimizer = Adam(secondPassLearningRate)
    trainingModel.compile(loss=create_triplet_loss(vectorSize), optimizer=optimizer)

    trainingModel.summary()
    trainingModel.count_params()

    trainingModel.fit_generator(
        generator=trainingGenerator,
        steps_per_epoch=stepsPerEpoch,
        epochs=secondPassEpochs,
        validation_data=testingGenerator,
        validation_steps=validationSteps,
        workers=workers,
        use_multiprocessing=True,
        max_queue_size=maxQueueSize,
        callbacks=[testNearestNeighbor, tensorBoardCallback]
    )


def generateTestImages(rotations):
    image = generatePillImage()
    rotated = {rotation: skimage.transform.rotate(image, angle=rotation, mode='constant', cval=1) for rotation in rotations}
    return image, rotated

globalMeasurementImages = []
globalMeasurementRotatedImages = []
def measureAccuracy(model):
    global globalMeasurementImages
    global globalMeasurementRotatedImages
    print("Measuring Accuracy")

    batchSize = 32
    datasetSizesToTest = [200, 1000, 5000]
    rotationsToTest = [5, 15, 25, 35, 45]
    maxDatasetSize = max(*datasetSizesToTest)
    printEvery = 250

    if len(globalMeasurementImages) < maxDatasetSize:
        print("    Generating test images")
        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(psutil.cpu_count() * 0.9)) as worker:
            for n in range(maxDatasetSize):
                futures.append(worker.submit(lambda: generateTestImages(rotationsToTest)))

            completedImages = 0
            for future in concurrent.futures.as_completed(futures):
                future.result()
                completedImages += 1

                if completedImages % printEvery == 0:
                    print(f"        Completed generating {completedImages} images.")

        globalMeasurementImages = [future.result()[0] for future in futures]
        globalMeasurementRotatedImages = [future.result()[1] for future in futures]

    originalVectors = {
        rotation: [] for rotation in rotationsToTest
    }
    rotatedVectors = {
        rotation: [] for rotation in rotationsToTest
    }

    print("    Computing vectors for images")
    for batch in range(int(math.ceil(maxDatasetSize / batchSize))):
        batchStart = batch * batchSize
        batchEnd = (batch + 1) * batchSize

        batchOriginals = globalMeasurementImages[batchStart:batchEnd]
        batchOriginalVectors = model.predict(numpy.array(batchOriginals))

        for rotation in rotationsToTest:
            batchRotated = globalMeasurementRotatedImages[batchStart:batchEnd][rotation]
            batchRotatedVectors = model.predict(numpy.array(batchRotated))

            for n in range(len(batchOriginals)):
                originalVector = batchOriginalVectors[n]
                rotatedVector = batchRotatedVectors[n]

                if numpy.any(numpy.isnan(originalVector)) or numpy.any(numpy.isnan(rotatedVector)) or numpy.any(numpy.isinf(originalVector)) or numpy.any(numpy.isinf(rotatedVector)):
                    print("        Error, NaN or infinity in final vector")
                else:
                    originalVectors[rotation].append(originalVector)
                    rotatedVectors[rotation].append(rotatedVector)

                if len(originalVectors) % printEvery == 0:
                    print(f"        Completed vectors for {len(originalVectors[[rotation]])} samples")

    print("    Measuring Final Accuracy")
    allAccuracies = []
    for rotation in rotationsToTest:
        if len(originalVectors[rotation]) == 0:
            print(f"        Error! Unable to measure accuracy for rotation {rotation}. All samples had bad vectors.")
            continue

        for datasetSize in datasetSizesToTest:
            print(f"        Measuring accuracy on {rotation} degree rotations with dataset size {datasetSize}")

            vectorIndexes = random.sample(range(len(originalVectors[rotation])), min(len(originalVectors[rotation]), datasetSize))

            origVectorsForTest = [originalVectors[rotation][index] for index in vectorIndexes]
            rotatedVectorsForTest = [rotatedVectors[rotation][index] for index in vectorIndexes]

            print(f"            Fitting the nearest neighbor model on {len(origVectorsForTest)} samples")
            nearestNeighborModel = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
            nearestNeighborModel.fit(origVectorsForTest)

            print(f"            Computing results of nearest neighbors model on rotated images.")
            distance, indices = nearestNeighborModel.kneighbors(rotatedVectorsForTest)

            correct = 0
            for n in range(len(origVectorsForTest)):
                if indices[n] == n:
                    correct += 1

            accuracy = float(correct) / float(len(origVectorsForTest))
            print(f"            Nearest Neighbor Accuracy on {rotation} degree rotations with {datasetSize} total dataset size: {accuracy}")
            allAccuracies.append(accuracy)

    if len(allAccuracies) == 0:
        return 0

    meanAccuracy = numpy.mean(allAccuracies)
    print(f"    Final Mean Accuracy {meanAccuracy}")
    return meanAccuracy

trainModel()
