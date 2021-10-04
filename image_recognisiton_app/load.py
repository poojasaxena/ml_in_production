"""
to load the pretrained model and json.
"""
import tensorflow as tf
from tensorflow.keras.models import model_from_json

def init():
    """Return the loaded model and graph"""
    try:
        with open('model/model.json','r', encoding="utf-8") as f_json:
            json = f_json.read()
    except OSError:
        print("model.json is not found")
    loaded_model=model_from_json(json)
    loaded_model.load_weights("model/model.h5")
    print("Weights are loaded from disk")
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #loaded_model.run_eagerly=True
    graph = tf.compat.v1.get_default_graph()
    return loaded_model, graph
