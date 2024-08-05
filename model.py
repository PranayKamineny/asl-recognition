import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.preprocessing.image import ImageDataGenerator

# DESIGN MODEL

model = Sequential()

# convolutional layer with 32 3x3 filters and pooling layer to reduce dimensions for next layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)))
model.add(MaxPooling2D((2, 2)))

# convolutional layer with 64 3x3 filters and pooling layer to reduce dimensions for next layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# convolutional layer with 128 3x3 filters and pooling layer to reduce dimensions for next layer
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# flatten remaining 2D vector into a 1D vector
model.add(Flatten())

# two fully connected layers with 512 neurons each and a dropout rate of 0.5 in both
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# output layer with 29 neurons with a softmax activation
model.add(Dense(29, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# TRAIN MODEL

# define data generators to convert images to grayscale and rescale pixel values from [0-255] to [0-1]
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# configure training data to flow from training directory and shuffle from each class with batch size of 32
train_generator = train_datagen.flow_from_directory(
    'archive/asl_alphabet_train',
    target_size=(200, 200),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# same configuration for validation dataset
val_generator = val_datagen.flow_from_directory(
    'archive/asl_alphabet_val',
    target_size=(200, 200),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10
)

# save the trained model
model.save('asl_hand_sign_recognizer.h5')