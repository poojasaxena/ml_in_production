import os

class Config(object):
    LOSS_FUNCTION   = "categorical_crossentropy"
    OPTIMIZER       = "adam"
    HOST_NAME       = "0.0.0.0"
    PORT_NUMBER     = 5000
    DEBUG_STATE     = 'True'
    CLASSES_NAME    = ['dog','cat']
    IMAGE_SIZE      = 200
    SAVED_MODEL_DIR = os.environ.get('DirForSavedModel')
    MAIL_USERNAME   = os.environ.get('MAIL_USERNAME')
    
class CnnConfig(Config):
    MODEL_JSON     = "model_dogcat_classifier_cnn_85acc_okt21.json"
    MODEL_H5       = "model_dogcat_classifier_cnn_85acc_okt21.h5"
    THRESHOLD      = 80
    
class AugmentationConfig(Config):
    MODEL_JSON     = "model_dogcat_classifier_cnn_augmentation_90acc_okt21.json"
    MODEL_H5       = "model_dogcat_classifier_cnn_augmentation_90acc_okt21.h5"
    THRESHOLD      = 70
    
class TransferLearningConfig(Config):
    MODEL_JSON     = "model_dogcat_classifier_vgg_tLearn_93acc_okt21.json"
    MODEL_H5       = "model_dogcat_classifier_vgg_tLearn_93acc_okt21.h5"
    IMAGE_SIZE     = 224
    THRESHOLD      = 80

class TransferLearningConfig95(Config):
    MODEL_JSON     = "model_dogcat_classifier_vgg_tLearn_95acc_okt21.json"
    MODEL_H5       = "model_dogcat_classifier_vgg_tLearn_95acc_okt21.h5"
    IMAGE_SIZE     = 224
    THRESHOLD      = 80
