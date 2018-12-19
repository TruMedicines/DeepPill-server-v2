from keras.applications import MobileNetV2, InceptionV3
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Reshape, Input, merge, Flatten, Subtract, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from keras.regularizers import l1, l2
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import csv
from scipy.misc import imsave, imread
import os
import sklearn.metrics
import numpy
import matplotlib.pyplot as plt
import os.path
import random
import sklearn.neighbors
import json
from pprint import pprint
from generate_data import generatePillImage
import skimage.transform
import random
import concurrent.futures

def trainModel():
    batchSize = 8
    batchNumber = 0
    minRotation = 5
    maxRotation = 30

    def generateBatch():
        nonlocal batchNumber
        while True:
            batchNumber += 1

            # Generate half the batch as negative examples, half the batch as positive examples
            inputs = []
            outputs = []

            # Generate negatives as just separate images
            for n in range(int(batchSize/2)):
                image1 = generatePillImage()
                image2 = generatePillImage()

                rotated1 = skimage.transform.rotate(image1, angle=random.uniform(minRotation, maxRotation), mode='constant', cval=1)
                rotated2 = skimage.transform.rotate(image2, angle=random.uniform(minRotation, maxRotation), mode='constant', cval=1)

                inputs.append(numpy.array([rotated1, rotated2]))
                outputs.append(0)

            # Generate positives as being the same image except rotated
            for n in range(int(batchSize/2)):
                image = generatePillImage()

                rotated1 = skimage.transform.rotate(image, angle=random.uniform(0, minRotation), mode='constant', cval=1)
                rotated2 = skimage.transform.rotate(image, angle=random.uniform(minRotation, maxRotation), mode='constant', cval=1)

                inputs.append(numpy.array([rotated1, rotated2]))
                outputs.append(1)

            yield numpy.array(inputs), numpy.array(outputs)


    trainingPrimary = Input((2, 256, 256, 3))
    predictPrimary = Input((256, 256, 3))

    imageNet = Sequential()
    imageNet.add(InceptionV3(include_top=False, pooling=None, input_shape=(256, 256, 3), weights=None))
    imageNet.add(Reshape([-1]))
    imageNet.add(BatchNormalization())
    imageNet.add(Dropout(0.5))
    imageNet.add(Dense(1024, activation='elu'))
    imageNet.add(BatchNormalization())
    imageNet.add(Dropout(0.5))
    imageNet.add(Dense(256, activation='tanh'))

    encoded_l = imageNet(Lambda(lambda x: x[:, 0])(trainingPrimary))
    encoded_r = imageNet(Lambda(lambda x: x[:, 1])(trainingPrimary))

    encoded_predict = imageNet(predictPrimary)

    # merge two encoded inputs with the l1 distance between them
    L1_distance = lambda x: K.abs(x[0] - x[1])

    both = Lambda(L1_distance, output_shape=lambda x: x[0])([encoded_l, encoded_r])

    prediction = Dense(1, activation='sigmoid')(both)
    trainingModel = Model(inputs=[trainingPrimary], outputs=prediction)
    predictionModel = Model(inputs=[predictPrimary], outputs=encoded_predict)

    optimizer = Adam(1e-4)
    trainingModel.compile(loss="binary_crossentropy", optimizer=optimizer)

    trainingModel.summary()
    trainingModel.count_params()
    predictionModel.summary()
    predictionModel.count_params()

    testingGenerator = generateBatch()
    trainingGenerator = generateBatch()

    def epochCallback(epoch, logs):
        predictionModel.compile(loss="mean_squared_error", optimizer=optimizer)
        measureAccuracy(predictionModel)

    testNearestNeighbor = LambdaCallback(on_epoch_end=epochCallback)

    trainingModel.fit_generator(
        generator=trainingGenerator,
        steps_per_epoch=1000,
        epochs=50,
        validation_data=testingGenerator,
        validation_steps=10,
        workers=7,
        use_multiprocessing=True,
        max_queue_size=50,
        callbacks=[testNearestNeighbor]
    )



globalMeasurementImageFutures = []
def measureAccuracy(model):
    global globalMeasurementImageFutures
    print("Measuring Accuracy on 1,000 images, 15 degree rotation")

    testSamples = 1000

    if len(globalMeasurementImageFutures) < testSamples:
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as worker:
            for n in range(testSamples):
                globalMeasurementImageFutures.append(worker.submit(generatePillImage))

    originalVectors = []
    rotatedVectors = []

    for n in range(testSamples):
        image = globalMeasurementImageFutures[n].result()
        rotated = skimage.transform.rotate(image, angle=15, mode='constant', cval=1)

        vectors = model.predict(numpy.array([image, rotated]))

        originalVectors.append(vectors[0])
        rotatedVectors.append(vectors[1])
        if (n+1) % 100 == 0:
            print(f"Completed vectors for {n+1} samples")

    print(f"Fitting the nearest neighbor model.")
    nearestNeighborModel = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
    nearestNeighborModel.fit(originalVectors)

    print(f"Computing results of nearest neighbors model.")
    distance, indices = nearestNeighborModel.kneighbors(rotatedVectors)

    correct = 0
    for n in range(testSamples):
        if indices[n] == n:
            correct += 1

    accuracy = float(correct) / float(testSamples)
    print("Nearest Neighbor Accuracy: ", accuracy)

trainModel()
