from pillmatch.model import PillRecognitionModel
import json
import sys
from pprint import pprint
from pillmatch.utilities import merge
from pillmatch.generated_dataset import GeneratedDataset
from pillmatch.loaded_dataset import LoadedDataset


def main():
    configFilePath = sys.argv[1]
    with open('default_parameters.json', 'rt') as file:
        defaultParameters = json.load(file)

    with open(configFilePath, 'rt') as file:
        parameters = json.load(file)

    mergedParameters = defaultParameters
    merge(parameters, mergedParameters)

    print("Testing model with the following parameters")
    pprint(mergedParameters)
    loaded = LoadedDataset(mergedParameters)
    loaded.loadImages()
    model = PillRecognitionModel(mergedParameters, GeneratedDataset(mergedParameters), loaded)

    model.loadModel(sys.argv[2])
    # model.measureAccuracy(model.model)
    model.finalMeasureAccuracy(model.model)


