import os

class Config(object):
    HOST_NAME      = "0.0.0.0"
    PORT_NUMBER    = 8000
    DEBUG_STATE    = 'False'
    SAVED_MODEL_DIR = os.environ.get('DirForSavedModel') + '/face_recognisition/'
    
class Developement(Config):
    MODEL_EMBEDDING       = "facenet_keras.h5"
    MODEL_CLASSIFICATION  = "svm_classifie_5_celebraties.pkl"
    THRESHOLD        = 25

class Production(Config):
    MODEL_EMBEDDING       = "facenet_keras.h5"
    MODEL_CLASSIFICATION  = "svm_classifie_5_celebraties.pkl"
    THRESHOLD        = 90

    
