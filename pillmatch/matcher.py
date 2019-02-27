from pillmatch.model import PillRecognitionModel
import json
import skimage.io
import skimage.transform
import sys
from pprint import pprint
from pillmatch.utilities import merge
from pillmatch.generated_dataset import GeneratedDataset
from pillmatch.loaded_dataset import LoadedDataset
import io
import sklearn
import numpy
import csv
from azure.storage.blob import BlockBlobService, PublicAccess


class Matcher:
    """ This class performs image matching for the endpoint."""

    def __init__(self, parameters):
        self.parameters = parameters

        self.dataset = LoadedDataset(self.parameters)
        self.model = PillRecognitionModel(self.parameters, GeneratedDataset(self.parameters), self.dataset)

        self.model.loadModel(f"model-best-weights.h5")


    def createVectorsFile(self):
        # Create the BlockBlockService that is used to call the Blob service for the storage account
        azureBlobStorage = BlockBlobService(account_name=self.parameters['azureStorageBucket'], account_key=self.parameters['azureStorageKey'],
                                            endpoint_suffix=self.parameters['azureEndpointSuffix'])

        fieldNames = ["imageId", "url"]
        for v in range(self.parameters['neuralNetwork']['vectorSize']):
            fieldNames.append(f'v{v}')

        with open('db-vectors.csv', 'wt') as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldNames))
            writer.writeheader()

            for imageId in range(len(LoadedDataset.rawImages)):
                rawImage = LoadedDataset.rawImages[imageId]

                buffer = io.BytesIO()
                skimage.io.imsave(buffer, rawImage)

                result = azureBlobStorage.create_blob_from_bytes('images', f"image-{imageId}.png", buffer.getvalue())

                images = self.dataset.getImageDBSet(imageId)

                vectors = self.model.model.predict(numpy.array(images))
                vectors = sklearn.preprocessing.normalize(vectors)

                for vector in vectors:
                    data = {
                        "imageId": imageId,
                        "url": f"{self.parameters['azureStorageUrl']}image-{imageId}.png"
                    }

                    for index, value in enumerate(vector):
                        data[f'v{index}'] = value

                    writer.writerow(data)


    def loadVectorsFile(self):
        ids = []
        vectors = []
        urls = {}

        with open('db-vectors.csv', 'rt') as f:
            reader = csv.DictReader(f)

            for row in reader:
                ids.append(row['imageId'])

                vector = []
                for v in range(self.parameters['neuralNetwork']['vectorSize']):
                    vector.append(row[f'v{v}'])

                vectors.append(vector)

                urls[row['imageId']] = row['url']

        return (ids, vectors, urls)

    def trainNearestNeighborModel(self):
        ids, vectors, urls = self.loadVectorsFile()

        self.nearestNeighborModel = sklearn.neighbors.NearestNeighbors(n_neighbors=3)
        self.nearestNeighborModel.fit(vectors)

        self.ids = ids
        self.urls = urls


    def findMatchForImage(self, image):
        # Crop the image to be a square, and resize to 224 pixels
        width, height = image.shape[1], image.shape[0]
        cropSize = int(min(width, height))
        startx = int((width - cropSize) / 2)
        starty = int((height - cropSize) / 2)

        image = image[starty:starty + cropSize, startx:startx + cropSize]

        image = skimage.transform.resize(image, (224, 224))[:,:,:3]

        # Now we make the query rotations
        testRotations = []
        for rotation in range(0, 360, self.parameters['finalTestQueryRotationIncrement']):
            rotated = skimage.transform.rotate(image, angle=rotation, mode='constant', cval=1)
            testRotations.append(rotated)

        testRotations = numpy.array(testRotations)

        vectors = self.model.model.predict(testRotations)
        vectors = sklearn.preprocessing.normalize(vectors)

        rotationDistances, rotationMatchingIndexes = self.nearestNeighborModel.kneighbors(vectors, n_neighbors=3)

        totalCounts = {}
        total = 0
        for distances, indexes in zip(rotationDistances, rotationMatchingIndexes):
            for distance, index in zip(distances, indexes):
                predictId = self.ids[index]
                totalCounts[predictId] = totalCounts.get(predictId, 0) + 1
                total +=1

        matches = [{
                "id": imageId,
                "url": self.urls[imageId],
                "confidence": totalCounts[imageId] / total
        } for imageId in totalCounts]

        matches = [match for match in matches if match['confidence'] > 0.20]
        matches = sorted(matches, key=lambda match: match['confidence'], reverse=True)

        return matches
