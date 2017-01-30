# Udacity Self Driving Car Course

### Project 3: Behavioral Cloning Project

#### Overview
This readme present my submission for Project 3: Behavioral cloning of Udacity Self Driving Car Nanodegree course.


#### Commands
###### Train the Network
```python model.py```

###### Behavioral Cloning of human driving
```python drive.py model.json```


#### Network Architecture
I've used [Nvidia] (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) model as starting point but after many tries, I've changed the model to get better result. Also I tried the [CommaAI](https://github.com/commaai/research/blob/master/train_steering_model.py) as well.

Hyperparameters:
* Learning rate: 0.001. I tried different learning rates and found 0.001 to be the most effective learning rate.
* Activation function: ELU
* Dropout keep prob: 0.5
* Optimizer: Adam. I chose the [Adam optimizer](https://keras.io/optimizers/#adam), because it's extremly robust and working well.
* Loss Function: MSE

The network consists of 9 layers, including a normalization layer, 4 convolutional layers and 4 fully connected layers along with a 50% dropout layer after each fully connected layer. The total number of parameters is 400,123.

I used [Keras' Lamda layer](https://keras.io/layers/core/#lambda) as input layer to normalize all input features on the GPU since it could do it much faster than on a CPU.

I've added a a dropout layer after each fully connected layer to avoid overfitting.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 20, 40, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 16, 36, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 8, 18, 24)     0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 4, 14, 36)     21636       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 2, 7, 36)      0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 2, 7, 48)      43248       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 1, 3, 48)      0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 1, 3, 64)      27712       maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 192)           0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1024)          197632      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1024)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           102500      dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 50)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
====================================================================================================
Total params: 400,123
Trainable params: 400,123
Non-trainable params: 0
```

#### Gathering Training Data
After training the network for many times, I found out that traning data is more important than network! IMO, Gathering trainging data was the hardest part of this project and took a lot of time. So I collected my own data in Self-Driving Car simulator on Mac OS using a playstation controller.

After recording data and training network many time, I saw the network fails on sharp turns. So I record some recovery data by driving the car towards end of lane with recording off and then driving it back (recovery) with recording on.

After collecting about 20k data, I've used 20% of that for validation and 80% for training purpose.

#### Training
I trained the model using the keras generator with batch size of 128 for 20 epochs. I trained the network on a g2.2xlarge EC2 instance, saved the model and weights persisted as model.json and model.h5 respectively,then scped model.json and model.h5 to my machine, then tested the model in autonomous mode using drive.py.

#### Augmentation Data (Pre-processing)
The augmentation algorithm consists the following steps:
* Considers left, center and right images at each iteration randomly
* Find mean of all steering angles and augment them
* Augment the brightness of camera image randomly
* Convert BGR to YUV colorspace
* Crops top 65 pixels (unnecessary for training) and bottom 20 pixels (remove the hood of the car)
* Blur image in order to smooth out some artifact
* Resize image to (20, 40)
* Flip the image about the vertical midline to simulate driving in opposite direction. The data collected had a lot of left turns, so to balance the left and right turns, images were flipped about the vertical axis randomly. The angles corresponding to flipped images were multplied by -1 to correct for steering in opposite direction.
