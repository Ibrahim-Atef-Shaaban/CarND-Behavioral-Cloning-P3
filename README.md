# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./Writeup_images/center_2021_01_31_15_33_32_120.jpg "Center Sample"
[image3]: ./Writeup_images/center_2021_01_31_16_45_14_586.jpg "Recovery Image"
[image4]: ./Writeup_images/center_2021_01_31_16_45_26_846.jpg "Recovery Image"
[image5]: ./Writeup_images/center_2021_01_31_16_45_32_893.jpg "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
### Note:
The speed was adjusted in the drive.py file, line 47 from 9 to 25

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 12 and 64 (model.py lines 110-124) 

The model includes RELU layers to introduce nonlinearity (used with every conv2d layer, lines 112, 114, 116, 118), and the data is normalized in the model using a Keras lambda layer (code line 110). 

Architecture summary is available in coming section.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 123). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 91-100). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and data from the simulator second track, I've also used the sample data provided by Udacity team for the network training. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use LeNet architecture, but then I indroduced some changes to it, changing the conv2d depths, applying max pooling, and finally adding the dropout layer before the final output layer, this didn't work smothly at the begining with only 2 Epochs, so I increased the number of epochs, which improved it, but still wasn't enough, the wining horse here was adding the lambda layer.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a slightly higher mean squared error on the validation set. This implied that the model was slightly overfitting. 

To combat the overfitting, I augmented the images fed to the network and also used data from the second track which helped to model to generalize more

The final step was to run the simulator to see how well the car was driving around track one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road even at 30 KPH.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes

Here is a summary on the model architecture

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 76, 316, 12)       912       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 38, 158, 12)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 34, 154, 24)       7224      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 17, 77, 24)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 75, 36)        7812      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 7, 37, 36)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 35, 64)         20800     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 2, 17, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2176)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 120)               261240    
_________________________________________________________________
dense_2 (Dense)              (None, 84)                10164     
_________________________________________________________________
dropout_1 (Dropout)          (None, 84)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 85        
=================================================================
Total params: 308,237
Trainable params: 308,237
Non-trainable params: 0

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would help the model to generalize and it doesn't have all input going the the left side as the rotation around the track needs

After the collection process, I had around 84000 number of data points. I then preprocessed this data by normalizing image values and cropping the image 60 pixels from top and 20 from bottom.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
