from pillmatch.matcher import Matcher
from pillmatch.utilities import merge
from pillmatch.pill_db import PillDB
from flask import Flask
from flask import request
import json
import numpy
import os
import skimage.io
import tempfile
import io
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


with open('default_parameters.json', 'rt') as file:
    defaultParameters = json.load(file)

with open("best.json", 'rt') as file:
    parameters = json.load(file)

mergedParameters = defaultParameters
merge(parameters, mergedParameters)

matcher = Matcher(mergedParameters)

# matcher.createVectorsFile()

matcher.trainNearestNeighborModel()

test = numpy.ones(shape=(224, 224, 3))
match = matcher.findMatchForImage(test) # this is used to warm up the server
print("Ready", flush=True)

db = PillDB()

@app.route("/ImageSearch", methods=['GET', 'POST'])
def imageMatch():
    print("received request")

    fileName = list(dict(request.files).keys())[0]

    data = request.files[fileName].read()

    db.addPill(data)

    buffer = io.BytesIO()
    buffer.write(data)
    buffer.seek(0)

    with tempfile.TemporaryDirectory() as tempDir:
        fileName = f"{tempDir}/temp.png"
        with open(fileName, "wb") as file:
            file.write(buffer.getvalue())

        image = skimage.io.imread(fileName)
        os.unlink(fileName)

    matches = matcher.findMatchForImage(image)

    for matchIndex, match in enumerate(matches):
        match['rank'] = matchIndex + 1

    data = [{
        "Imageurl": match['url'],
        "Name": f"Pill - {match['id']}",
        "Description": "",
        "Id": match['id'],
        "CreatedOn": "",
        "Percentage": match['rank'],
        "Total": len(matcher.ids)
    } for match in matches]

    return json.dumps(data)

@app.route("/PillCoordinates", methods=['GET'])
def pillCoordinates():
    return json.dumps({"pills": db.getPills()[-100:]})

def main():
    app.run(host="192.168.1.17")
