import json
import math
import os
import random

import cv2
import numpy as np
import pandas as pd
from scipy import misc

import tensorflow as tf
from sklearn.cross_validation import train_test_split
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, ELU
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from scipy import misc


class DataItem:
    """
    Define each the data sample
    Features:
        center : Path to CENTER camera image
        left   : Path to LEFT camera image
        right  : Path to RIGHT camera image
        steering : Steering angle in range -1 and 1
    """

    def __init__(self, data):
        self.data = data
        self.steering_angle = round(float(data['steering']), 4)

        left = data['left'].strip()
        center = data['center'].strip()
        right = data['right'].strip()

        left, center, right = [(os.path.join(os.path.dirname(__file__), './data/IMG', os.path.split(file_path)[1])) for file_path in (left, center, right)]

        self.left_image_path = left
        self.center_image_path = center
        self.right_image_path = right

    def __str__(self):
        output = []
        output.append('Left camera path: {}'.format(self.left_image_path))
        output.append('Center camera path: {}'.format(self.center_image_path))
        output.append('Right camera path: {}'.format(self.right_image_path))
        output.append('Steering angle: {}'.format(self.steering_angle))
        return '\n'.join(output)

def preprocess_image(image, output_shape):
    """
    Steps:
        1. Convert BGR to YUV colorspace
        2. Crops top 65 pixels and bottom 20 pixels
        3. Blur image
        4. Resize image
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    shape = image.shape
    image = image[65:140, 0:320]

    kernel_size = 5
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    image = cv2.resize(image, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA)

    return image


def batch_generator(X, y, label, num_epochs, batch_size, output_shape):
    """
    Batch generator used for feeding batch to model as training and valiation data
    Steps:
        * Consider left, center and right images based on a random int number
        * Calculate mean of all steering angles and augment camera steering samples (if needed)
        * Preprocess image
        * Randomly flip the image and steering angle 50% of time
    """

    epoch_index = 0
    num_data = len(X)
    batch_size  = min(num_data, batch_size)
    batch_count = int(math.ceil(num_data / batch_size))

    print('Batch generating for {}... Number of data:{}'.format(label,num_data))

    while True:
        for i in range(batch_count):
            X_batch = []
            y_batch = []

            start_i = epoch_index
            epoch_index += batch_size

            if epoch_index >= num_data:
                print('Reshuffling {}...'.format(label))
                shuffled_data = np.arange(num_data)
                np.random.shuffle(shuffled_data)
                X = X[shuffled_data]
                y = y[shuffled_data]
                start_i = 0
                epoch_index = batch_size

            end_i = min(epoch_index, num_data)

            y_mean = np.mean(y)
            threshold = abs(y_mean) * 0.01
            l_threshold = y_mean - threshold
            r_threshold = y_mean + threshold

            for j in range(start_i, end_i):
                data = X[j]
                steering_angle = y[j]

                # Augmenting the steering angle
                if steering_angle < l_threshold:
                    rnd = np.random.randint(3)

                    if rnd == 0:
                        image_path = data.right_image_path
                        angle_coefficient = 3.0
                    elif rnd == 1:
                        image_path = data.right_image_path
                        angle_coefficient = 2.
                    elif rnd == 2:
                        image_path = data.center_image_path
                        angle_coefficient = 1.5
                    else:
                        image_path = data.center_image_path
                        angle_coefficient = 1.

                if steering_angle > r_threshold:
                    rnd = np.random.randint(3)

                    if rnd == 0:
                        image_path = data.left_image_path
                        angle_coefficient = 3.0
                    elif rnd == 1:
                        image_path = data.left_image_path
                        angle_coefficient = 2.
                    elif rnd == 2:
                        image_path = data.center_image_path
                        angle_coefficient = 1.5
                    else:
                        image_path = data.center_image_path
                        angle_coefficient = 1.
                else:
                    image_path = data.center_image_path
                    angle_coefficient = 1.

                steering_angle *= angle_coefficient
                image_array = cv2.imread(image_path)

                # Preprocess the image
                image = preprocess_image(image_array, output_shape=output_shape)

                # Flip image 
                if random.random() > 0.5:
                    X_batch.append(cv2.flip(image, 1))
                    y_batch.append(-steering_angle)
                else:
                    X_batch.append(image)
                    y_batch.append(steering_angle)

            yield np.array(X_batch), np.array(y_batch)


def build_model(input_shape):
    learning_rate = 0.001
    keep_prob = 0.5
    activation = "elu"

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape, output_shape=output_shape))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation=activation))
    model.add(Flatten())
    model.add(Dropout(keep_prob))
    model.add(Dense(1024, activation=activation))
    model.add(Dropout(keep_prob))
    model.add(Dense(100, activation=activation))
    model.add(Dropout(keep_prob))
    model.add(Dense(50, activation=activation))
    model.add(Dropout(keep_prob))
    model.add(Dense(10, activation=activation))
    model.add(Dense(1, init='normal', activation=activation))

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    print('\nNetwork Architecture: ')
    model.summary()

    return model

# Hyper parameters
num_epochs = 16
batch_size = 128
input_shape = (20, 40, 3)
output_shape = (20, 40, 3)
validation_split_percentage = 0.2

model_file_name = 'model.json'
weights_file_name = 'model.h5'

# Step 1: Load dataset 
DRIVING_LOG_PATH = './data/driving_log.csv'
X_train = []
y_train = []

if os.path.isfile(DRIVING_LOG_PATH):
    dataframe = pd.read_csv(DRIVING_LOG_PATH)
    for index, data in dataframe.iterrows():
        data_item = DataItem(data=data)
        X_train.append(data_item)
        y_train.append(data_item.steering_angle)

## split data into test and validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=validation_split_percentage,
    random_state=0)

X_train = np.array(X_train)
y_train = np.array(y_train, dtype=np.float32) 
X_val = np.array(X_val)
y_val = np.array(y_val, dtype=np.float32)

if len(X_train) > 0:
    print('Input data:')
    print('Number of input data: ', len(X_train))
    center_camera_view = cv2.imread(X_train[0].center_image_path)
    print('Center camera view shape: {}\n'.format(center_camera_view.shape))
    print(X_train[0])


# Step 2: Define Model
model = build_model(input_shape)

# Step 3: Fit Model
train_data  = batch_generator(X=X_train, y=y_train, label='train set', num_epochs=num_epochs, batch_size=batch_size, output_shape=output_shape)
validation_data = batch_generator(X=X_val, y=y_val, label='validation set', num_epochs=num_epochs, batch_size=batch_size, output_shape=output_shape)
history = model.fit_generator(
    train_data,
    nb_epoch=num_epochs, 
    samples_per_epoch=len(X_train),
    nb_val_samples=len(X_val),
    verbose=2, 
    validation_data=validation_data)

print(history.history)

# Step 4: Save Model
model.save_weights(weights_file_name)
with open(model_file_name, 'w') as outfile:
    json.dump(model.to_json(), outfile)

