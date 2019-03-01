from pillmatch.matcher import Matcher
from pillmatch.utilities import merge
from pillmatch.pill_db import PillDB
from flask import Flask
from flask import request, send_from_directory
import json
import numpy
import os
import skimage.io
import tempfile
import io
from flask_cors import CORS
clientFolder = '/home/bradley/eb-pill-match/pillmatch/client'
app = Flask(__name__, static_folder=clientFolder)
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


@app.route('/static/<path:path>', methods=['GET'])
def send_client_static(path):
    return send_from_directory(f'{clientFolder}/static', path)

@app.route('/themes/<path:path>', methods=['GET'])
def send_client_themes(path):
    return send_from_directory(f'{clientFolder}/themes', path)

@app.route('/locales/<path:path>', methods=['GET'])
def send_client_locales(path):
    return send_from_directory(f'{clientFolder}/locales', path)

@app.route('/img/<path:path>', methods=['GET'])
def send_client_img(path):
    return send_from_directory(f'{clientFolder}/img', path)

@app.route('/', methods=['GET'])
def root():
    return app.send_static_file(f'index.html')

@app.route('/index.html', methods=['GET'])
def root3():
    return app.send_static_file(f'index.html')

@app.route('/asset-manifest.json', methods=['GET'])
def asset():
    return app.send_static_file(f'asset-manifest.json')

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return app.send_static_file(f'favicon.ico')

@app.route('/manifest.json', methods=['GET'])
def manifest():
    return app.send_static_file(f'manifest.json')

@app.route('/service-worker.js', methods=['GET'])
def servieworker():
    return app.send_static_file(f'service-worker.js', methods=['GET'])

@app.route('/precache-manifest.95cd47c065c9cb29362506bd58b89271', methods=['GET'])
def precache():
    return app.send_static_file(f'precache-manifest.95cd47c065c9cb29362506bd58b89271')


def main():
    app.run(host="192.168.1.17")
