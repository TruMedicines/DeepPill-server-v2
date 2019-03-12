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
import skimage.color
import scipy.signal
import skimage.io
from pillmatch import textures
import cv2
import gc
from skimage.filters import threshold_otsu, threshold_local


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
            # iaa.AddToHueAndSaturation((self.params["trainingAugmentation"]["minHueShift"], self.params["trainingAugmentation"]["maxHueShift"])),
            iaa.AdditiveGaussianNoise(scale=self.params["trainingAugmentation"]["gaussianNoise"] * 255),
            # iaa.CoarseDropout(p=self.params['trainingAugmentation']['coarseDropoutProbability'], size_percent=self.params['trainingAugmentation']['coarseDropoutSize'])
        ])


        augmentations = []
        images = self._getRawPillImages(self.params['neuralNetwork']['augmentationsPerImage'], imageId)
        for n, image in enumerate(images):
            anchor = image
            anchor = skimage.transform.resize(anchor, (self.params['imageWidth'], self.params['imageHeight'], 3), mode='reflect', anti_aliasing=True)

            if self.params['preprocessing']['threshold']['enabled'] == 'false':
                # Apply custom hue transform
                hueShift = skimage.color.rgb2hsv(anchor) * 255.0
                hueShift[:, :, 0] = hueShift[:, :, 0] + random.randint(-50, 50)
                hueShift[:, :, 0] = numpy.mod(hueShift[:, :, 0], 255)
                hueShift[:, :, 1] = hueShift[:, :, 1] + random.randint(-50, 50)
                hueShift[:, :, 1] = numpy.minimum(hueShift[:, :, 1], 255)
                anchor = skimage.color.hsv2rgb(hueShift / 255.0)

                anchor = self.addShadow(anchor)

            anchor = self._applyPreprocessing(anchor)

            rotationDirection = random.choice([-1, +1])
            anchorAugmented = skimage.transform.rotate(anchor, angle=random.uniform(self.currentMinRotation / 2, self.currentMaxRotation / 2) * rotationDirection, mode='constant', cval=0)
            if self.params['preprocessing']['threshold']['enabled'] == 'false':
                anchorAugmented = numpy.maximum(numpy.zeros_like(anchorAugmented), anchorAugmented)
                anchorAugmented = numpy.minimum(numpy.ones_like(anchorAugmented), anchorAugmented)

                anchorAugmented = noiseAugmentation.augment_images(numpy.array([anchorAugmented]) * 255.0)[0] / 255.0

            augmentations.append(anchorAugmented)

        result = numpy.array(augmentations, copy=True)

        if imageId % 10 == 0:
            gc.collect()

        return result

    def getRotationTestingImageSet(self, imageId):
        """
            This method returns a set of pill images that are to be used for basic rotation testing.
            There is an original 'anchor' image that is put into the db, and then multiple
            'testing' images with increasing amounts of rotation.

            :return: (baseImage, rotatedImages)
        """
        image = self._getRawPillImages(1, imageId)[0]

        testRotations = self.params['rotationsToTest']

        image = skimage.transform.resize(image, (self.params['imageWidth'], self.params['imageHeight'], 3), mode='reflect', anti_aliasing=True)

        testRotated = {rotation: self._applyPreprocessing(skimage.transform.rotate(image, angle=rotation, mode='constant', cval=1)) for rotation in testRotations}

        return self._applyPreprocessing(image), testRotated

    def getFinalTestingImageSet(self, imageId):
        self.setRotationParams(0, 360)
        self.params["generateAugmentation"]["minRotation"] = 0
        self.params["generateAugmentation"]["maxRotation"] = 360
        rawImages = self._getRawPillImages(self.params["finalTestDBAugmentations"] + self.params['finalTestAugmentationsPerImage'], imageId, applyAugmentationsAfter=self.params["finalTestDBAugmentations"])

        dbImages = []
        for dbImage in rawImages[:self.params["finalTestDBAugmentations"]]:
            dbImage = skimage.transform.resize(dbImage, (self.params['imageWidth'], self.params['imageHeight'], 3), mode='reflect', anti_aliasing=True)
            # for shrinkage in [32, 64, 96]:
            #     newSize = (int(self.params['imageWidth']-shrinkage), int(self.params['imageHeight']-shrinkage), 3)
            #     shrunk = skimage.transform.resize(dbImage, newSize, mode='reflect', anti_aliasing=True)
            #     width_pad = int(shrinkage/2)
            #     height_pad = int(shrinkage/2)
            #     padded = skimage.util.pad(shrunk, ((width_pad, width_pad), (height_pad, height_pad), (0, 0)), mode="constant", constant_values=1)

            for rotation in range(0, 360, self.params['finalTestRotationIncrement']):
                dbImages.append(self._applyPreprocessing(skimage.transform.rotate(dbImage, angle=rotation, mode='constant', cval=1)))

        testImages = []
        for image in rawImages[self.params["finalTestDBAugmentations"]:]:
            image = skimage.transform.resize(image, (self.params['imageWidth'], self.params['imageHeight'], 3), mode='reflect', anti_aliasing=True)

            # Now we make the query rotations
            testRotations = []
            for rotation in range(0, 360, self.params['finalTestQueryRotationIncrement']):
                rotated = skimage.transform.rotate(image, angle=rotation, mode='constant', cval=1)
                testRotations.append(self._applyPreprocessing(rotated))
            testImages.append(testRotations)

        data = (imageId, dbImages, testImages)
        return data

    def getImageDBSet(self, imageId):
        self.setRotationParams(0, 360)
        rawImages = self._getRawPillImages(self.params["finalTestDBAugmentations"], imageId, applyAugmentationsAfter=self.params["finalTestDBAugmentations"])

        dbImages = []
        for dbImage in rawImages:
            dbImage = skimage.transform.resize(dbImage, (self.params['imageWidth'], self.params['imageHeight'], 3), mode='reflect', anti_aliasing=True)
            for rotation in range(0, 360, self.params['finalTestRotationIncrement']):
                dbImages.append(self._applyPreprocessing(skimage.transform.rotate(dbImage, angle=rotation, mode='constant', cval=1)))

        return dbImages


    def _getRawPillImages(self, count, imageId, applyAugmentationsAfter=0):
        """ This method is meant to be implemented by subclasses. It should return multiple available
            raw pill images, as it would appear if a user took it from their camera.
        """
        pass


    def extractCircle(self, image):
        grey = cv2.cvtColor(numpy.array(image * 255.0, dtype=numpy.uint8), cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1, 500, param1=4, param2=1, minRadius=60, maxRadius=112)

        if circles is None:
            return image

        cropCircle = circles[0][0]

        mask = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=numpy.uint8)
        rr, cc = skimage.draw.circle(cropCircle[1], cropCircle[0], cropCircle[2],
                                     shape=(image.shape[0], image.shape[1]))
        mask[rr, cc, :] = 1
        image = image * mask

        cropTop = max(0, int(cropCircle[1] - cropCircle[2]))
        cropBottom = min(int(cropCircle[1] + cropCircle[2]), image.shape[1])
        cropLeft = max(0, int(cropCircle[0] - cropCircle[2]))
        cropRight = min(int(cropCircle[0] + cropCircle[2]), image.shape[0])

        cropped = image[cropTop:cropBottom, cropLeft:cropRight]

        image = skimage.transform.resize(cropped, (image.shape[0], image.shape[1], 3), mode='reflect',
                                         anti_aliasing=True)

        del cropped, mask

        return image
    def _applyPreprocessing(self, image):
        """ This method should apply preprocessing to the given image. This includes convert-to-grayscale,
            edge detection, and hough-circles to try and crop out the circle.
        """
        if self.params['preprocessing']['detectCircle'] == 'true':
            image = self.extractCircle(image)

        if self.params['preprocessing']['edgeDetection']['enabled'] == 'true':
            grey = cv2.cvtColor(numpy.array(image * 255.0, dtype=numpy.uint8), cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(grey, int(self.params['preprocessing']['edgeDetection']['threshold']/2), self.params['preprocessing']['edgeDetection']['threshold'], 5)
            image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            image = numpy.array(image, dtype=numpy.float32) / 255.0

        if self.params['preprocessing']['threshold']['enabled'] == 'true':
            grey = skimage.color.rgb2gray(image)
            thresh = threshold_local(grey, block_size=31, offset=0.05)
            binary = grey <= thresh

            binary = scipy.signal.medfilt(binary, kernel_size=(3, 3))

            image[:, :, 0] = binary
            image[:, :, 1] = binary
            image[:, :, 2] = binary

        return image


    def _applyTrainingAugmentation(self, image):
        """ This method should apply training augmentations to the given pill image."""
        pass


    def _applyTestingAugmentation(self, image):
        """ This method should apply testing augmentations to the given pill image"""
        pass

    def generateExamples(self, save=True, show=False):
        for n in range(6):
            images = self._getRawPillImages(5, n)

            for imageIndex, image in enumerate(images):
                if save:
                    skimage.io.imsave('raw-' + str(n) + "-" + str(imageIndex) + ".png", image)
                if show:
                    plt.imshow(image, interpolation='lanczos')
                    plt.show()

        for n in range(6):
            images = self.getTrainingImageSet(n)

            for imageIndex, image in enumerate(images):
                if save:
                    skimage.io.imsave('training-' + str(n) + "-" + str(imageIndex) + ".png", image)
                if show:
                    plt.imshow(image, interpolation='lanczos')
                    plt.show()

        for n in range(6):
            base, rotations = self.getRotationTestingImageSet(n)

            skimage.io.imsave('testing-' + str(n) + "-base.png", base)

            for imageIndex, image in enumerate(rotations.values()):
                if save:
                    skimage.io.imsave('testing-' + str(n) + "-" + str(imageIndex) + ".png", image)
                if show:
                    plt.imshow(image, interpolation='lanczos')
                    plt.show()

    def generateShadowCoordinates(self, imshape, no_of_shadows=1):
        vertices_list = []
        for index in range(no_of_shadows):
            vertex = []
            for dimensions in range(numpy.random.randint(3, 4)):  ## Dimensionality of the shadow polygon
                vertex.append((imshape[1] * numpy.random.uniform(), imshape[0] // 3 + imshape[0] * numpy.random.uniform()))
            vertices = numpy.array([vertex], dtype=numpy.int32)  ## single shadow vertices 
            vertices_list.append(vertices)
        return vertices_list  ## List of shadow vertices
    
    
    def addShadow(self, image):
        imshape = image.shape
        # Primary shadow / lightness effects.
        vertices_list = self.generateShadowCoordinates(imshape, 3)
        for vertices in vertices_list:
            mask = numpy.zeros_like(image)
            cv2.fillPoly(mask, vertices, 255)
            if random.uniform(0, 1) < 0.5:
                adjust = random.uniform(0, 0.75)
                image[:, :, :][mask[:, :, 0] == 255] = image[:, :, :][mask[:, :, 0] == 255] * (1.0 - adjust) + numpy.ones_like(image[:, :, :][mask[:, :, 0] == 255]) * adjust
            else:
                adjust = random.uniform(0, 0.25)
                image[:, :, :][mask[:, :, 0] == 255] = image[:, :, :][mask[:, :, 0] == 255] * (1.0 - adjust) + numpy.zeros_like(image[:, :, :][mask[:, :, 0] == 255]) * adjust

        # A hue shift, simulates observed color shift in some images
        image_hsv = skimage.color.rgb2hsv(image)
        vertices_list = self.generateShadowCoordinates(imshape, 1)
        for vertices in vertices_list:
            mask = numpy.zeros_like(image)
            cv2.fillPoly(mask, vertices, 255)
            adjust = random.uniform(-0.3, 0.3)
            image_hsv[:, :, 0][mask[:, :, 0] == 255] = numpy.mod(image[:, :, 0][mask[:, :, 0] == 255] + adjust, 1)

        image = skimage.color.hsv2rgb(image_hsv)

        return image