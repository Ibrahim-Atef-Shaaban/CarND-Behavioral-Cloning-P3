# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:44:27 2021

@author: Ibrahim SHAABAN
"""
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D as conv2d
from keras.layers.pooling import MaxPool2D

#Global image(input) and measurements(Labels) lists
images = []
measurements = []

#Reading the csv lines
def getCSV_data(data_loc):
    with open (data_loc+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        file_lines = []
        for line in reader:
            file_lines.append(line)
    return file_lines

def getTrain_data(data_loc):
    global images, measurements
    lines = getCSV_data(data_loc)
    # Since there is a slight difference between the data provided and the current
    # simulator generation, the first line had to be removed, so I'm starting from
    # the second line
    if "sample" in data_loc:
        for line in lines[1:]:
            for idx in range (3):

                #Adding a correction to the left and right images
                if idx == 0:#Center image, no correction needed
                    correction = 0
                elif idx == 1:#Left image, positive correction needed
                    correction = 0.2
                elif idx == 2:#Right image, negative correction needed
                    correction = -0.2
                    
                #Getting the image path
                source_path = line[idx]
                filename = source_path.split('/')[-1]
                current_path = data_loc + 'IMG/' + filename
                
                #Reading the image and converting from BGR to RGB
                image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
                
                #Saving data to the global lists
                images.append(image)
                measurement = float(line[3])
                measurements.append(measurement)
                #Flipping the image and the mesurement
                images.append(cv2.flip(image,1))
                measurements.append(measurement*-1.0)
    else:
        for line in lines:
            for idx in range (3):
                
                #Adding a correction to the left and right images
                if idx == 0:#Center image, no correction needed
                    correction = 0
                elif idx == 1:#Left image, positive correction needed
                    correction = 0.2
                elif idx == 2:#Right image, negative correction needed
                    correction = -0.2
                    
                #Getting the image path
                source_path = line[idx]
                filename = source_path.split('\\')[-1]
                current_path = data_loc + 'IMG/' + filename
                
                #Reading the image and converting from BGR to RGB
                image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
                
                #Saving data to the global lists
                images.append(image)
                measurement = float(line[3]) + correction
                measurements.append(measurement) 
                #Flipping the image and the mesurement
                images.append(cv2.flip(image,1))
                measurements.append(measurement*-1.0)
            
# Loading the sample data, provided by udacity
getTrain_data('./data_sample/')
print("Finished sample driving data!")

# Loading my data, which is:
# 1- Two laps at the center in the 1st track
# 2- One lap getting from sides and moving to the middle on the first track
# 3- One lap in the second track
getTrain_data('./data/')
print("Finished my driving data!")


#Assigning the training data and their labels to numpy arrays, and deleting
# the images and measurements from local variables to save memory
x_train = np.array(images)
del images
y_train = np.array(measurements)
del measurements

#Building the model pipeline
model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((60,20), (0,0))))
model.add(conv2d(12,5,5, activation = 'relu'))
model.add(MaxPool2D())
model.add(conv2d(24,5,5, activation = 'relu'))
model.add(MaxPool2D())
model.add(conv2d(36,3,3, activation = 'relu'))
model.add(MaxPool2D())
model.add(conv2d(64,3,3, activation = 'relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dropout(0.35))
model.add(Dense(1))
model.compile(loss="mse", optimizer = "adam")
model.fit(x_train, y_train, validation_split=0.2, shuffle = True, epochs=5,
          batch_size=32)
#Show the model summary and saving the model
model.summary()
model.save('model.h5')
