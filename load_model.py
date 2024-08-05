import cv2
import os
import numpy as np
import tensorflow as tf
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import matplotlib.pyplot as plt

# load saved model
model = keras.models.load_model('asl_hand_sign_recognizer.h5')
model.summary()

# TEST MODEL

# define test data generator with same parameters
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# same configuration for test dataset
test_generator = test_datagen.flow_from_directory(
    'archive/asl_alphabet_test',
    target_size=(200,200),
    color_mode='grayscale',
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# get predictions on test set from the model
predictions = model.predict(test_generator)

# get class labels and print results
class_labels = list(test_generator.class_indices.keys())
for i, filename in enumerate(test_generator.filenames):
    predicted_class = class_labels[np.argmax(predictions[i])]
    print(f'Image: {filename}, Predicted Class: {predicted_class}')