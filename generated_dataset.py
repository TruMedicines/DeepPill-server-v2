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

class GeneratedDataset(Dataset):
    """ This class is responsible for creating the generated dataset for the core algorithm. """

    def __init__(self, params):
        Dataset.__init__(self, params)

        self.params = params

    def _getRawPillImages(self, count):
        mainAugmentation = iaa.Sequential([
            iaa.Affine(
                scale=(self.params["generateAugmentation"]["minScale"], self.params["generateAugmentation"]["maxScale"]),
                translate_percent=(self.params["generateAugmentation"]["minTranslate"], self.params["generateAugmentation"]["maxTranslate"]),
                shear=(self.params["generateAugmentation"]["minShear"], self.params["generateAugmentation"]["maxShear"]),
                cval=255),
            iaa.GaussianBlur(sigma=(self.params["generateAugmentation"]["minGaussianBlur"], self.params["generateAugmentation"]["maxGaussianBlur"])),
            iaa.MotionBlur(k=(self.params["generateAugmentation"]["minMotionBlur"], self.params["generateAugmentation"]["maxMotionBlur"]))
        ])

        secondAugmentation = iaa.Sequential([
            iaa.GammaContrast(gamma=(self.params["generateAugmentation"]["minContrast"], self.params["generateAugmentation"]["maxContrast"])),
            iaa.Add((self.params["generateAugmentation"]["minBrightness"], self.params["generateAugmentation"]["maxBrightness"]))
        ])

        anchor = self._generateRawPillImage()

        augmentations = []
        for n in range(count):
            rotationDirection = random.choice([-1, +1])
            anchorAugmented = skimage.transform.rotate(anchor, angle=random.uniform(self.params["generateAugmentation"]["minRotation"] / 2, self.params["generateAugmentation"]["maxRotation"] / 2) * rotationDirection,
                                                       mode='constant', cval=1)
            anchorAugmented = mainAugmentation.augment_images(numpy.array([anchorAugmented]) * 255.0)[0]
            anchorAugmented = anchorAugmented / 255.0

            anchorWhiteMask = sklearn.preprocessing.binarize(numpy.mean(anchorAugmented, axis=2), threshold=0.99, copy=False)
            anchorWhiteMask = numpy.repeat(anchorWhiteMask[:, :, numpy.newaxis], 3, axis=2)

            randomTexture = random.choice(textures.textures)
            anchorAugmented = anchorWhiteMask * randomTexture + (1.0 - anchorWhiteMask) * anchorAugmented

            anchorAugmented = secondAugmentation.augment_images(numpy.array([anchorAugmented]) * 255.0)[0]
            anchorAugmented = anchorAugmented / 255.0

            augmentations.append(anchorAugmented)
        return augmentations

    def _generatePerlinNoise2d(self, shape, res):
        def f(t):
            return 6* t ** 5 - 15 * t ** 4 + 10 * t ** 3
    
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        # noise = np.random.poisson(lam=2.5, size=(res[0] + 1, res[1] + 1))
        # noise = ((noise - np.min(noise)) / (np.max(noise) - np.min(noise)))
        # noise = np.random.normal(loc=0.5, scale=0.2, size=(res[0] + 1, res[1] + 1))
        noise = np.random.rand(res[0] + 1, res[1] + 1)
        angles = 2 * np.pi * noise
    
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

    def _generateRandomRectangles(self, shape):
        data = np.zeros(shape)
    
        width = shape[0]
        height = shape[0]
    
        for n in range(50):
            randomRotation = random.uniform(0, 360)
            center = (random.randint(0, width), random.randint(0, height))
    
            rectWidth = random.randint(10, 35)
            rectHeight = random.randint(10, 20)
    
            rect = (center, (rectWidth, rectHeight), randomRotation)
    
            randomAlpha = random.uniform(0.2, 0.6)
    
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rectImage = data.copy()
            cv2.fillPoly(rectImage, [box], (1, 1, 1))
    
            cv2.addWeighted(rectImage, randomAlpha, data, 1 - randomAlpha, 0, data)
    
        # plt.imshow(data, interpolation='lanczos')
        # plt.show()
        return data


    def _cropCircle(self, imageData):
        width = imageData.shape[0]
        height = imageData.shape[1]
    
        circleCenterX = int(width/2)
        circleCenterY = int(height/2)
    
        circleRadius = int(min(width, height)/2)
    
        mask = np.zeros((width, height, 3), dtype=np.uint8)
    
        rr, cc = skimage.draw.circle(circleCenterY, circleCenterX, circleRadius, shape=(width, height))
        mask[rr, cc, :] = 1
    
        out = imageData * mask

        white = 1.0 * (1.0 - mask)
    
        imageData = out + white
        return imageData

    def _generateRawPillImage(self):
        """ This method is meant to be implemented by subclasses. It should return a single raw pill image.
            If this is the real pill dataset, these would be the high-res pill images taken when the pill
            was manufactured.
        """
        width = 256
        height = 256

        splotchPatternWidth = 384
        splotchPatternHeight = 384

        np.random.seed(int(time.time() * 1000) % (2 ** 30))
        splotchMaskPattern = self._generatePerlinNoise2d((splotchPatternWidth, splotchPatternHeight), (16, 16))
        rectanglePattern = self._generateRandomRectangles((splotchPatternWidth, splotchPatternHeight))
        splotchMaskPattern = np.maximum(splotchMaskPattern * 0.8, rectanglePattern)
        splotchMaskPattern = sklearn.preprocessing.binarize(splotchMaskPattern, threshold=0.40, copy=False)

        # splotchMaskPattern = scipy.signal.medfilt(splotchMaskPattern, kernel_size=(13, 13))
        # splotchMaskPattern = sklearn.preprocessing.binarize(splotchMaskPattern, threshold=0.60, copy=False)
        #
        # splotchMaskPattern = scipy.signal.medfilt(splotchMaskPattern, kernel_size=(13, 13))
        # splotchMaskPattern = sklearn.preprocessing.binarize(splotchMaskPattern, threshold=0.60, copy=False)

        splotchMask = np.zeros((width, height, 1))
        splotchMask[:, :, 0] = splotchMaskPattern[:width, :height]
        splotchMask = scipy.ndimage.filters.gaussian_filter(splotchMask, 0.6, order=0)

        backgroundTexture = np.zeros((width, height, 3))
        backgroundPattern = np.zeros((width, height, 1))
        backgroundPattern[:, :, 0] = self._generatePerlinNoise2d((width, height), (16, 16))
        backgroundColor1 = np.zeros((width, height, 3))
        backgroundColor2 = np.zeros((width, height, 3))
        backgroundColor1[:, :] = np.array((242, 246, 238)) / 256.0
        backgroundColor2[:, :] = np.array((242, 250, 230)) / 256.0
        backgroundTexture[:, :] = backgroundPattern * backgroundColor1 + (1.0 - backgroundPattern) * backgroundColor2

        backgroundPattern2 = np.zeros((width, height, 1))
        backgroundPattern2[:, :, 0] = self._generatePerlinNoise2d((width, height), (64, 64))
        backgroundTexture[:, :] += (backgroundPattern2 - 1.0) * 0.02

        splotchTexture = np.zeros((width, height, 3))
        splotchPattern = np.zeros((width, height, 1))
        splotchPattern[:, :, 0] = self._generatePerlinNoise2d((width, height), (16, 16))
        splotchColor1 = np.zeros((width, height, 3))
        splotchColor2 = np.zeros((width, height, 3))
        splotchColor1[:, :] = np.array((181, 67, 10)) / 256.0
        splotchColor2[:, :] = np.array((170, 12, 12)) / 256.0
        splotchTexture[:, :] = splotchPattern * splotchColor1 + (1.0 - splotchPattern) * splotchColor2

        imageData = splotchMask * splotchTexture + (1.0 - splotchMask) * backgroundTexture
        imageData[:, :] += np.random.normal(loc=0.0, scale=0.01, size=(width, height, 3))

        imageData = self._cropCircle(imageData)

        return imageData
