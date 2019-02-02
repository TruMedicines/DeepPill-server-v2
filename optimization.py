from model import PillRecognitionModel
import json
import sys
from pprint import pprint
from utilities import merge
from generated_dataset import GeneratedDataset

def trainRound1Optimization(parameters):
    with open('default_parameters.json', 'rt') as file:
        defaultParameters = json.load(file)

    mergedParameters = defaultParameters
    merge(parameters, mergedParameters)

    if '$budget' in mergedParameters:
        mergedParameters['neuralNetwork']['epochs'] = int(mergedParameters['$budget'])

    print("Training model with the following parameters")
    pprint(mergedParameters)
    dataset = GeneratedDataset(mergedParameters)
    model = PillRecognitionModel(mergedParameters, dataset)
    accuracy = model.trainModel()
    return {"loss": 1.0 - accuracy, "status": "ok"}



