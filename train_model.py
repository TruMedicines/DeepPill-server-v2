from model import PillRecognitionModel
import json
import sys
from pprint import pprint
from utilities import merge
from generated_dataset import GeneratedDataset

configFilePath = sys.argv[1]
with open('default_parameters.json', 'rt') as file:
    defaultParameters = json.load(file)

with open(configFilePath, 'rt') as file:
    parameters = json.load(file)

mergedParameters = defaultParameters
merge(parameters, mergedParameters)

print("Training model with the following parameters")
pprint(mergedParameters)
dataset = GeneratedDataset(mergedParameters)
model = PillRecognitionModel(mergedParameters, dataset)
model.trainModel()


