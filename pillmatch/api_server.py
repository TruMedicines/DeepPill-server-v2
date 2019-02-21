from flask import Flask
from flask import request
import json
app = Flask(__name__)

@app.route("/ImageSearch", methods=['GET', 'POST'])
def imageMatch():


    print("received request")

    data = {
        "Imageurl": "",
        "Name": "",
        "Description": "",
        "Id": 0,
        "CreatedOn": "",
        "Percentage": "55"
    }

    return json.dumps([data])

