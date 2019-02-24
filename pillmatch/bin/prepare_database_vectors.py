import json
import skimage.io
import sys
from pprint import pprint
from pillmatch.utilities import merge
import io
import sklearn
import numpy
import csv
from pillmatch.matcher import Matcher
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

    matcher = Matcher(mergedParameters)

    matcher.createVectorsFile()



