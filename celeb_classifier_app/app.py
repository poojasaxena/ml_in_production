#!/usr/bin/env python3.7

"""
Flask file to classify 5 celebraties
Author : Pooja SAXENA
Date   : 27 Oktober 2021
Place  : Hamburg
"""
import re
import sys
import os
import base64
import json
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from helper_class import labels_encoder, extract_face, get_128vectorEmbedding, load_labels
from tensorflow.compat.v1 import get_default_graph
from keras.models import load_model
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import platform
print(f"python version: {platform.python_version()}")

app = Flask(__name__)

## For Config, either use this
from configmodule import *
app.config.from_object('configmodule.Developement')
## or this
# app.config.from_pyfile('./config/testing_enviroment.cfg')
## or this
# json, check readme for more details

@app.route('/')
def index_view():
    """return template index.html"""
    return render_template('index.html')

def convert_image(img_data_uploaded):
    """To Convert the uploaded image"""
    imgstr = re.search(b'base64,(.*)',img_data_uploaded).group(1)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))

def load_models():
    # load embedding model
    facenet_model = load_model(app.config["SAVED_MODEL_DIR"]+app.config['MODEL_EMBEDDING'])
    # prediction for the face
    classification_model  = pickle.load(open(app.config['SAVED_MODEL_DIR']+app.config['MODEL_CLASSIFICATION'], "rb"))
    return facenet_model,classification_model

@app.route('/api/predict/',methods=['GET','POST'])
def predict():
    """Predict function to predict an uploaded image """
    graph_mode= get_default_graph()
    img_data = request.get_data()
    convert_image(img_data)
    image_pixels = extract_face('output.png')

    ## model calling
    facenet_model, classifiation_model  = load_models()
    emb_image    = get_128vectorEmbedding(facenet_model, image_pixels)
    print(">> embedded image shape:", emb_image.shape)
    
    with graph_mode.as_default():        
        samples_image = np.expand_dims(emb_image, axis=0)
        yhat_class    = classifiation_model.predict(samples_image)
        yhat_prob     = classifiation_model.predict_proba(samples_image)
        
        # setup encoder for test images    
        labels  = load_labels()
        label_encoder, label_y_class  = labels_encoder(labels)

        # get name
        class_index       = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
    
        if class_probability >= app.config["THRESHOLD"]:
            predicted_name = label_encoder.inverse_transform(yhat_class)[0]
        else:
            predicted_name = 'Unknown'

        prediction = {'class':predicted_name, 'confidence_level':round(class_probability,2)}
        print(prediction)
        return jsonify(prediction)

if __name__ == '__main__':
    app.debug = app.config["DEBUG_STATE"]
    app.run(host=app.config["HOST_NAME"], port=app.config["PORT_NUMBER"])


