"""
Basic Flask file to upload and predict an uploaded image.
"""
import re
import sys
import os
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
sys.path.append(os.path.abspath("./model"))

app = Flask(__name__)

@app.route('/')
def index_view():
    """return template index.html"""
    return render_template('index.html')

def convert_image(img_data_uploaded):
    """To Convert the uploaded image"""
    imgstr = re.search(b'base64,(.*)',img_data_uploaded).group(1)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/predict/',methods=['GET','POST'])
def predict():
    """Predict function to predict an uploaded image """
    graph_mode=tf.compat.v1.get_default_graph()
    img_data = request.get_data()
    convert_image(img_data)
    img_image = image.load_img('output.png', target_size=(150, 150))
    img_tensor = image.img_to_array(img_image)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    with graph_mode.as_default():
        json_file = open('model/model.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model/model.h5")
        loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

        pred = loaded_model.predict(img_tensor)
        pred_class = "dog" if pred[0]>0.5 else "cat"
        confidence = round((1-pred[0][0])*100 if pred_class=='cat' else pred[0][0]*100, 3)
        prediction = {'class':pred_class, 'confidence_level':confidence}
        print(prediction)
        return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True, port=8000)

