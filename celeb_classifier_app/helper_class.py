#!/usr/bin/env python3.7

import os
import numpy as np
from PIL import Image  
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder

def extract_labels(directory):
    y = []
    for subdir in os.listdir(directory):
        path = directory + subdir + '/'
        if not os.path.isdir(path):
            continue
        range_dir = [subpath for subpath in os.listdir(path)]
        labels = [subdir for _ in range(len(range_dir))]
        y.extend(labels)
    return np.asarray(y)
    
def load_labels():
    base_dir = os.environ.get('DirForImages')+'/5-celebrity-faces-dataset/'
    dataset_path = os.path.join(base_dir, 'train/')
    labels = extract_labels(dataset_path)
    return labels

def labels_encoder(labels):
    """provide one hot encoding of incoming labels"""
    out_encoder = LabelEncoder()
    out_encoder.fit(labels)
    labels = out_encoder.transform(labels)
    return out_encoder, labels
    
def extract_face(filename, required_size=(160, 160)):
    # pre-processing on file image
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)

    # create the mtcnn detector using default weights
    detector = MTCNN()
    face = detector.detect_faces(pixels)
    
    x1, y1, width, height = face[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array
    
    
def get_128vectorEmbedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    face_pixels_normalized = (face_pixels - face_pixels.mean()) / face_pixels.std()
    samples = np.expand_dims(face_pixels_normalized, axis=0)
    yhat = model.predict(samples)
    return yhat[0]
