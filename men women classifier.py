# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:15:23 2020

@author: Gaurav
"""
"""Building CNN"""
# This is initializaation of nural network
from keras.models import Sequential
#First step os cnn Convolutional 2d 
from keras.layers import Convolution2D
# Do 2nd step Maxpooling
from keras.layers import MaxPooling2D
#3 step Flattering (Is used to reduce no of input node) to the vector
from keras.layers import Flatten
# Used to add full conection 
from keras.layers import Dense

# Initializing CNN
classifier=Sequential()

# Step 1 -  Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

# Step 2 MaxPooling (Reduceing the size)
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2nd layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 Flatning
classifier.add(Flatten())

# Step 4- Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compilling
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

# Fitting our CNN to our images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

trainning_set= train_datagen.flow_from_directory(
        'E:/All Data Set/men-women-classification - Copy/Train',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'E:/All Data Set/men-women-classification - Copy/test',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        trainning_set,
        steps_per_epoch=2810,
        epochs=2,
        validation_data=test_set,
        validation_steps=499)


