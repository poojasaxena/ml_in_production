import re
import sys 
import os
import base64

import numpy as np
import tensorflow.keras.models

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from load import *

sys.path.append(os.path.abspath("./model"))

global graph, model

model, graph = init()
app = Flask(__name__)


@app.route('/')
def index_view():
    return render_template('index.html')

def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/predict/',methods=['GET','POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    json_file = open('model/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
        #load weights into new model
    loaded_model.load_weights("model/model.h5")
    print("Loaded Model from disk")
        #compile and evaluate loaded model
    loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        # perform the prediction
   # out = loaded_model.predict(img)

    #x = image.load_img('output.png', target_size=(150, 150))
    #x = image.load_img(imgData, target_size=(150, 150))
    #x = image.img_to_array(x)  
    #x = np.expand_dims(x, axis=0)       
    #x /= 255.
    #with graph.as_default():
    #    out = model.predict(x)
    #    print(out)
    #    print(np.where(out[0] < 0.5, 0 , 1))
    #    response = np.where(out[0] < 0.5, 0 , 1)#np.array_str(np.argmax(out,axis=1))
    #    return response	

if __name__ == '__main__':
    app.run(debug=True, port=8000)

