import os

class Config(object):
    HOST_NAME             = "0.0.0.0"
    PORT_NUMBER           = 8000
    DEBUG_STATE           = 'True'
    SAVED_MODEL_DIR       = os.environ.get('DirForSavedModel') + '/face_recognisition/'
    LOCAL_DATASET_DIR     = os.environ.get('DirForLocalDataset') + '/6-celebrity-faces-dataset/'
    MODEL_CLASSIFICATION  = "svm_classifie_6_celebraties.pkl"    

class Developement(Config):
    MODEL_EMBEDDING       = "facenet_keras.h5"
    THRESHOLD             = 50

class Production(Config):
    MODEL_EMBEDDING       = "facenet_keras.h5"
    THRESHOLD             = 90

    
