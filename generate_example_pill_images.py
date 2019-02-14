import generated_dataset
import loaded_dataset
from model import PillRecognitionModel
import json
import sys
from pprint import pprint
from utilities import merge

configFilePath = sys.argv[1]
with open('default_parameters.json', 'rt') as file:
    defaultParameters = json.load(file)

with open(configFilePath, 'rt') as file:
    parameters = json.load(file)

mergedParameters = defaultParameters
merge(parameters, mergedParameters)


dataset = loaded_dataset.LoadedDataset(mergedParameters)

dataset.generateExamples()
