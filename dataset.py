import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import skimage
import skimage.transform
import random
import pickle
import numpy
import sklearn
import os
import sklearn.preprocessing
import textures
import cv2

class Dataset:
    """ This class is responsible for providing the dataset which is used to train and test the core algorithm. """

    def __init__(self, params):
        self.params = params

        self.currentMinRotation = params['trainingAugmentation']['minRotation']
        self.currentMaxRotation = params['trainingAugmentation']['maxRotation']



    def setRotationParams(self, currentMinRotation, currentMaxRotation):
        self.currentMinRotation = currentMinRotation
        self.currentMaxRotation = currentMaxRotation

    def getTrainingImageSet(self, imageId):
        """ This method is used to return a single set of pill images that are of the same pill.

            These images have augmentation applied to them.


            :returns [numpy.array] An array of pill images that are of the same pill
        """
        noiseAugmentation = iaa.Sequential([
            iaa.PiecewiseAffine(self.params["trainingAugmentation"]["piecewiseAffine"]),
            iaa.AdditiveGaussianNoise(scale=self.params["trainingAugmentation"]["gaussianNoise"] * 255)
        ])

        augmentations = []
        images = self._getRawPillImages(self.params['neuralNetwork']['augmentationsPerImage'])
        for n, image in enumerate(images):
            anchor = self._applyPreprocessing(image)

            rotationDirection = random.choice([-1, +1])
            anchorAugmented = skimage.transform.rotate(anchor, angle=random.uniform(self.currentMinRotation / 2, self.currentMaxRotation / 2) * rotationDirection, mode='constant', cval=0)

            anchorAugmented = noiseAugmentation.augment_images(numpy.array([anchorAugmented]) * 255.0)[0] / 255.0

            anchorAugmented = skimage.transform.resize(anchorAugmented, (self.params['imageWidth'], self.params['imageHeight'], 3), mode='reflect', anti_aliasing=True)

            augmentations.append(anchorAugmented)

        result = numpy.array(augmentations)

        return result

    def getRotationTestingImageSet(self):
        """
            This method returns a set of pill images that are to be used for basic rotation testing.
            There is an original 'anchor' image that is put into the db, and then multiple
            'testing' images with increasing amounts of rotation.

            :return: (baseImage, rotatedImages)
        """
        image = self._getRawPillImages(1)[0]

        testRotations = self.params['rotationsToTest']

        image = skimage.transform.resize(image, (self.params['imageWidth'], self.params['imageHeight'], 3), mode='reflect', anti_aliasing=True)

        testRotated = {rotation: self._applyPreprocessing(skimage.transform.rotate(image, angle=rotation, mode='constant', cval=1)) for rotation in testRotations}

        return image, testRotated


    def getFinalTestingImageSet(self):
        """
            This method returns a set of pill images that are to be used for final testing. There is an
            original 'anchor' image that is put into the db, and then multiple 'testing' images
            with various levels of augmentation.

            :return:
        """

        pass


    def _getRawPillImages(self, count):
        """ This method is meant to be implemented by subclasses. It should return multiple available
            raw pill images, as it would appear if a user took it from their camera.
        """
        pass


    def _applyPreprocessing(self, image):
        """ This method should apply preprocessing to the given image. This includes convert-to-grayscale,
            edge detection, and hough-circles to try and crop out the circle.
        """
        if self.params['preprocessing']['detectCircle'] == 'true':
            grey = cv2.cvtColor(numpy.array(image * 255.0, dtype=numpy.uint8), cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1, 500, param1=4, param2=1, minRadius=60, maxRadius=112)

            cropCircle = circles[0][0]

            mask = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=numpy.uint8)
            rr, cc = skimage.draw.circle(cropCircle[1], cropCircle[0], cropCircle[2], shape=(image.shape[0], image.shape[1]))
            mask[rr, cc, :] = 1
            image = image * mask

            cropTop = max(0, int(cropCircle[1] - cropCircle[2]))
            cropBottom = min(int(cropCircle[1] + cropCircle[2]), image.shape[1])
            cropLeft = max(0, int(cropCircle[0] - cropCircle[2]))
            cropRight = min(int(cropCircle[0] + cropCircle[2]), image.shape[0])

            cropped = image[cropTop:cropBottom, cropLeft:cropRight]

            image = skimage.transform.resize(cropped, (image.shape[0], image.shape[1], 3), mode='reflect', anti_aliasing=True)

            del cropped, mask

        if self.params['preprocessing']['edgeDetection']['enabled'] == 'true':
            grey = cv2.cvtColor(numpy.array(image * 255.0, dtype=numpy.uint8), cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(grey, int(self.params['preprocessing']['edgeDetection']['threshold']/2), self.params['preprocessing']['edgeDetection']['threshold'], 5)
            image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            image = numpy.array(image, dtype=numpy.float32) / 255.0

        return image


    def _applyTrainingAugmentation(self, image):
        """ This method should apply training augmentations to the given pill image."""
        pass


    def _applyTestingAugmentation(self, image):
        """ This method should apply testing augmentations to the given pill image"""
        pass

    def generateExamples(self, save=True, show=False):
        for n in range(5):
            images = self._getRawPillImages(5)

            for imageIndex, image in enumerate(images):
                plt.imshow(image, interpolation='lanczos')
                if save:
                    plt.savefig('raw-' + str(n) + "-" + str(imageIndex) + ".png")
                if show:
                    plt.show()

        for n in range(3):
            images = self.getTrainingImageSet(n)

            for imageIndex, image in enumerate(images):
                plt.imshow(image, interpolation='lanczos')
                if save:
                    plt.savefig('training-' + str(n) + "-" + str(imageIndex) + ".png")
                if show:
                    plt.show()

        for n in range(3):
            base, rotations = self.getRotationTestingImageSet()

            plt.imshow(base, interpolation='lanczos')
            plt.savefig('testing-' + str(n) + "-base.png")

            for imageIndex, image in enumerate(rotations.values()):
                plt.imshow(image, interpolation='lanczos')
                if save:
                    plt.savefig('testing-' + str(n) + "-" + str(imageIndex) + ".png")
                if show:
                    plt.show()
