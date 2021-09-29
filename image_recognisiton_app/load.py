import numpy as np
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras import backend


def init():
    json_file = open('model/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load weights into new model
    loaded_model.load_weights("model/model.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)
    #out = loaded_model.predict(img)
    #print(out)
    #print(class_names[np.argmax(out)])
    # convert the response to a string
    #response = class_names[np.argmax(out)]
    #return str(response)
    graph = tf.compat.v1.get_default_graph()
    return loaded_model,graph
