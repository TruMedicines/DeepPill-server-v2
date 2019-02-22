from pillmatch.model import PillRecognitionModel
import json
import skimage.io
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


def main():
    configFilePath = sys.argv[1]
    with open('default_parameters.json', 'rt') as file:
        defaultParameters = json.load(file)

    with open(configFilePath, 'rt') as file:
        parameters = json.load(file)

    mergedParameters = defaultParameters
    merge(parameters, mergedParameters)

    print("Training model with the following parameters")
    pprint(mergedParameters)
    dataset = LoadedDataset(mergedParameters)
    model = PillRecognitionModel(mergedParameters, GeneratedDataset(mergedParameters), dataset)

    model.loadModel(f"model-best-weights.h5")

    # Create the BlockBlockService that is used to call the Blob service for the storage account
    azureBlobStorage = BlockBlobService(account_name=mergedParameters['azureStorageBucket'], account_key=mergedParameters['azureStorageKey'])

    with open('db-vectors.csv', 'wt') as f:
        writer = csv.DictWriter(f, fieldnames=list(dbVectors[0].keys()))
        writer.writeheader()

        for imageId in range(len(LoadedDataset.rawImages)):
            rawImage = LoadedDataset.rawImages[imageId]

            buffer = io.BytesIO()
            skimage.io.imsave(buffer, rawImage)

            result = azureBlobStorage.create_blob_from_bytes('images', f"image-{imageId}.png", buffer.getvalue())

            images = dataset.getFinalTestingImageSet(imageId)

            vectors = model.predict(numpy.array(images))
            vectors = sklearn.preprocessing.normalize(vectors)

            for vector in vectors:
                data  = {
                    "imageId": imageId,
                    "url": f"{mergedParameters['azureStorageUrl']}image-{imageId}.png"
                }

                for index, value in enumerate(vector):
                    data[f'v{index}'] = value

                writer.writerow(data)




