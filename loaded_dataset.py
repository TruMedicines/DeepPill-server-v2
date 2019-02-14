import numpy as np
import skimage
import scipy
import scipy.stats
import scipy.misc
import sklearn
import time
import random
import math
import numpy
from pprint import pprint
from imgaug import augmenters as iaa
import sklearn.preprocessing
import os
import skimage.transform
import skimage.draw
import skimage.io
import matplotlib.pyplot as plt
import cv2
from dataset import Dataset
import textures
import skimage.io


class LoadedDataset(Dataset):
    """ This class is responsible for creating the generated dataset for the core algorithm. """

    def __init__(self, params):
        Dataset.__init__(self, params)

        self.params = params

        self.rawImages = []

        for file in os.listdir('test-images'):
            self.rawImages.append(skimage.transform.resize(cv2.imread(f"test-images/{file}"), output_shape=(256, 256)))

    def _getRawPillImages(self, count, imageId):
        mainAugmentation = iaa.Sequential([
            iaa.Affine(
                scale=(self.params["loadAugmentation"]["minScale"], self.params["loadAugmentation"]["maxScale"]),
                translate_percent=(self.params["loadAugmentation"]["minTranslate"], self.params["loadAugmentation"]["maxTranslate"]),
                shear=(self.params["loadAugmentation"]["minShear"], self.params["loadAugmentation"]["maxShear"]),
                cval=255)
        ])

        secondAugmentation = iaa.Sequential([
            iaa.GammaContrast(gamma=(self.params["loadAugmentation"]["minContrast"], self.params["loadAugmentation"]["maxContrast"])),
            iaa.Add((self.params["loadAugmentation"]["minBrightness"], self.params["loadAugmentation"]["maxBrightness"])),
            iaa.GaussianBlur(sigma=(self.params["loadAugmentation"]["minGaussianBlur"], self.params["loadAugmentation"]["maxGaussianBlur"])),
            iaa.MotionBlur(k=(self.params["loadAugmentation"]["minMotionBlur"], self.params["loadAugmentation"]["maxMotionBlur"]))
        ])

        anchor = self.rawImages[imageId % len(self.rawImages)]

        augmentations = []
        for n in range(count):
            rotationDirection = random.choice([-1, +1])
            anchorAugmented = skimage.transform.rotate(anchor, angle=random.uniform(self.params["generateAugmentation"]["minRotation"] / 2,
                                                                                    self.params["generateAugmentation"]["maxRotation"] / 2) * rotationDirection,
                                                       mode='constant', cval=1)
            anchorAugmented = mainAugmentation.augment_images(numpy.array([anchorAugmented]) * 255.0)[0]
            anchorAugmented = anchorAugmented / 255.0
            anchorAugmented = numpy.maximum(0, anchorAugmented)

            anchorWhiteMask = sklearn.preprocessing.binarize(numpy.mean(anchorAugmented, axis=2), threshold=0.99, copy=False)
            anchorWhiteMask = numpy.repeat(anchorWhiteMask[:, :, numpy.newaxis], 3, axis=2)

            randomTexture = random.choice(textures.textures)
            anchorAugmented = anchorWhiteMask * randomTexture + (1.0 - anchorWhiteMask) * anchorAugmented

            anchorAugmented = secondAugmentation.augment_images(numpy.array([anchorAugmented]) * 255.0)[0]
            anchorAugmented = anchorAugmented / 255.0

            augmentations.append(anchorAugmented)
        return augmentations
