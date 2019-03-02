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
from pillmatch.dataset import Dataset
from pillmatch import textures
import skimage.io
import concurrent.futures

class LoadedDataset(Dataset):
    """ This class is responsible for creating the generated dataset for the core algorithm. """

    def __init__(self, params):
        Dataset.__init__(self, params)

        self.params = params

    @staticmethod
    def loadImages():
        images = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for file in sorted(os.listdir('test-images')):
                if file.endswith("-cropped.png"):
                    continue

                fileName = f"test-images/{file}"

                futures.append(executor.submit(LoadedDataset.loadAndConvertImageFile, fileName))

            images = [future.result() for future in concurrent.futures.as_completed(futures) if future.result() is not None]

        random.shuffle(images)

        LoadedDataset.rawImages = images

    @staticmethod
    def loadAndConvertImageFile(fileName):
        image = skimage.io.imread(fileName)
        # image = numpy.flip(image, 0)
        # image = numpy.flip(image, 1)

        width = image.shape[1]
        height = image.shape[0]

        image = skimage.transform.resize(image, output_shape=(256, int(float(width) / float(height) * 256)))

        width = image.shape[1]

        grey = cv2.cvtColor(numpy.array(image * 255.0, dtype=numpy.uint8), cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1, width, param1=50, param2=25, minRadius=int(width / 5), maxRadius=int(width / 2))

        if circles is None:
            print(f"Error on image {file}")
            return None

        cropCircle = circles[0][0]

        mask = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=numpy.uint8)
        rr, cc = skimage.draw.circle(cropCircle[1], cropCircle[0], cropCircle[2], shape=(image.shape[0], image.shape[1]))
        mask[rr, cc, :] = 1
        image = image * mask
        image = image + (1.0 - mask)

        radius = cropCircle[2] * 1.1

        cropTop = max(0, int(cropCircle[1] - radius))
        cropBottom = min(int(cropCircle[1] + radius), image.shape[0])
        cropLeft = max(0, int(cropCircle[0] - radius))
        cropRight = min(int(cropCircle[0] + radius), image.shape[1])

        cropped = image[cropTop:cropBottom, cropLeft:cropRight]
        cropped = skimage.transform.resize(cropped, output_shape=(256, 256))

        return cropped

    def _getRawPillImages(self, count, imageId, applyAugmentations=True):
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

        anchor = LoadedDataset.rawImages[imageId % len(self.rawImages)]
        anchor = skimage.transform.rotate(anchor, angle=random.uniform(0, 360), mode='constant', cval=1)

        augmentations = []
        for n in range(count):
            rotationDirection = random.choice([-1, +1])
            anchorAugmented = skimage.transform.rotate(anchor, angle=random.uniform(self.params["loadAugmentation"]["minRotation"] / 2,
                                                                                    self.params["loadAugmentation"]["maxRotation"] / 2) * rotationDirection,
                                                       mode='constant', cval=1)
            if applyAugmentations:
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
