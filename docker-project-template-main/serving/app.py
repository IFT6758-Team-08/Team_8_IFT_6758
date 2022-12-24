"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
from utils_ import download_model

# import ift6758


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
MODEL_REPO = os.path.join('.','models')
global model
global model_requested
app = Flask(__name__)

logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """

    global model
    global model_requested

    print('Inside first request')
    app.logger.info('Starting...')
    default_model = 'xgb3'
    model_requested = default_model

    model_path = os.path.join(MODEL_REPO, default_model+"1.0.2" +'.joblib')
    print(model_path)
    print(os.path.exists(model_path))
    if os.path.exists(model_path):
        app.logger.info("Default model exists in local model repository...")
        pass
    else:
        app.logger.info("Downloading default model from comet...")
        print("here")
        download_model("rachel98","ift-6758-team-8","DPA8v9aBQumK4h2GAkMp6RA5d","xgb3","1.0.2")
        os.rename( os.path.join(MODEL_REPO, default_model+'.joblib'), model_path)
        print("downloaded")
    model = joblib.load(model_path)
    print(model)
    app.logger.info("Loaded the Default Model: " + model_path)
    pass


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data
    with open(LOG_FILE, 'r') as logger:
        resp = logger.readlines()
    return jsonify(resp) # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    global model
    global model_requested
    # Get POST json data
    json = request.get_json()
    print(json)
    app.logger.info(json)

    # TODO: check to see if the model you are querying for is already downloaded
    model_requested = json["model"]
    # print(model_requested)
    workspace = json["workspace"]
    version = json["version"]

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    model_path = os.path.join(MODEL_REPO, model_requested + version+'.joblib')
    
    if os.path.exists(model_path):
        app.logger.info("Requested model exists in local model repository...")
    else:
        app.logger.info('Requested model needs to be downloaded from comet')
        download_model(workspace, "ift-6758-team-8", "DPA8v9aBQumK4h2GAkMp6RA5d", model_requested, version)
        os.rename(os.path.join(MODEL_REPO, model_requested +'.joblib'), model_path)
    print("loading model ", model_path)
    model = joblib.load(model_path)
    # app.logger.info()
    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    # raise NotImplementedError("TODO: implement this endpoint")

    response = "Requested model successfully loaded!"

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    print('Inside Predict endpoint')
    json = request.get_json()
    print("json is loaded")
    print("the model is ", model_requested)
    # print("json is ", json)
    X_test = pd.read_json(json, orient='table', convert_dates=False)
    print("Json is oriented")
    print(X_test)
    y_pred = model.predict(X_test.values)
    print(y_pred)
    y_pred_prob = model.predict_proba(X_test.values)[:,1]
    print("y pred prob", y_pred_prob)
    # TODO:
    # raise NotImplementedError("TODO: implement this enpdoint")
    
    response = pd.DataFrame({'predictions':y_pred, 'pred_proba':y_pred_prob}, columns=['predictions','pred_proba']).to_json()

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!
