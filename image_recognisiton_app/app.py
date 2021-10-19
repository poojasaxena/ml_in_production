"""
Basic Flask file to upload and predict an uploaded image.
Author : Pooja SAXENA
Date   : 02 Oktober 2021
Place  : Hamburg
"""
import re
import sys
import os
import base64
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.compat.v1 import get_default_graph
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

app = Flask(__name__)

## For Config, either use this
from configmodule import *
app.config.from_object('configmodule.AugmentationConfig')
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

#@app.route('/predict/',methods=['GET','POST'])
@app.route('/api/predict/',methods=['GET','POST'])
def predict():
    """Predict function to predict an uploaded image """
    graph_mode= get_default_graph()
    img_data = request.get_data()
    convert_image(img_data)
    img_image = image.load_img('output.png', target_size=(app.config['IMAGE_SIZE'], app.config['IMAGE_SIZE']))
    img_tensor = image.img_to_array(img_image)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    with graph_mode.as_default():
        try:
            with open(app.config["SAVED_MODEL_DIR"] +"/dog_cat_classifier/" + app.config["MODEL_JSON"],'r', encoding="utf-8") as f_json:
                json_model = f_json.read()
        except OSError:
            print(f"{f_json} is not found")
        loaded_model = model_from_json(json_model)
        loaded_model.load_weights(app.config["SAVED_MODEL_DIR"] +"/dog_cat_classifier/"+ app.config["MODEL_H5"])
        loaded_model.compile(loss=app.config["LOSS_FUNCTION"],optimizer=app.config["OPTIMIZER"],metrics=['accuracy'])

        pred = loaded_model.predict(img_tensor)
        pred_class = app.config["CLASSES_NAME"][0] if pred[0]>0.5 else app.config["CLASSES_NAME"][1]
        confidence = round((1-pred[0][0])*100 if pred_class==app.config["CLASSES_NAME"][1] else pred[0][0]*100, 3)
        prediction = {'class':pred_class, 'confidence_level':confidence}
        print(prediction)
        return jsonify(prediction)

if __name__ == '__main__':
    app.debug = app.config["DEBUG_STATE"]
    app.run(host=app.config["HOST_NAME"], port=app.config["PORT_NUMBER"])


