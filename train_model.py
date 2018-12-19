from keras.applications import MobileNetV2, InceptionV3
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Reshape, Input, merge, Flatten, Subtract, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
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
import json
from pprint import pprint
from generate_data import generatePillImage
import skimage.transform
import random

def trainModel():
    batchSize = 32
    batchNumber = 0

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

                rotated1 = skimage.transform.rotate(image1, angle=random.uniform(-30, +30), mode='constant', cval=1)
                rotated2 = skimage.transform.rotate(image2, angle=random.uniform(-30, +30), mode='constant', cval=1)

                inputs.append(numpy.array([rotated1, rotated2]))
                outputs.append(0)

            # Generate positives as being the same image except rotated
            for n in range(int(batchSize/2)):
                image = generatePillImage()

                rotated1 = skimage.transform.rotate(image, angle=random.uniform(-30, +30), mode='constant', cval=1)
                rotated2 = skimage.transform.rotate(image, angle=random.uniform(-30, +30), mode='constant', cval=1)

                inputs.append(numpy.array([rotated1, rotated2]))
                outputs.append(1)

            yield numpy.array(inputs), numpy.array(outputs)


    input_shape = (2, 256, 256, 3)
    primary = Input(input_shape)

    imageNet = Sequential()
    imageNet.add(InceptionV3(include_top=False, pooling=None, input_shape=(256, 256, 3), weights=None))
    imageNet.add(Reshape([-1]))
    imageNet.add(BatchNormalization())
    imageNet.add(Dropout(0.5))
    imageNet.add(Dense(1024, activation='elu'))
    imageNet.add(BatchNormalization())
    imageNet.add(Dropout(0.5))
    imageNet.add(Dense(256, activation='tanh'))

    encoded_l = imageNet(Lambda(lambda x: x[:, 0])(primary))
    encoded_r = imageNet(Lambda(lambda x: x[:, 1])(primary))

    # merge two encoded inputs with the l1 distance between them
    L1_distance = lambda x: K.abs(x[0] - x[1])

    both = Lambda(L1_distance, output_shape=lambda x: x[0])([encoded_l, encoded_r])

    prediction = Dense(1, activation='sigmoid')(both)
    model = Model(inputs=[primary], outputs=prediction)

    optimizer = Adam(1e-4)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    model.summary()

    model.count_params()

    testingGenerator = generateBatch()
    trainingGenerator = generateBatch()

    model.fit_generator(
        generator=trainingGenerator,
        steps_per_epoch=100,
        epochs=50,
        validation_data=testingGenerator,
        validation_steps=10,
        workers=7,
        use_multiprocessing=True,
        max_queue_size=50
    )



trainModel()
