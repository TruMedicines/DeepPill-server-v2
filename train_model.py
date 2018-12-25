from model import PillRecognitionModel
import json
import sys
from pprint import pprint

configFilePath = sys.argv[1]
with open('default_parameters.json', 'rt') as file:
    defaultParameters = json.load(file)

with open(configFilePath, 'rt') as file:
    parameters = json.load(file)

for k in defaultParameters:
    if k not in parameters:
        parameters[k] = defaultParameters[k]

print("Training model with the following parameters")
pprint(parameters)
model = PillRecognitionModel(parameters)
model.trainModel()


