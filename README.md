# Udacity Self Driving Car Course

### Project 3: Behavioral Cloning Project

#### Network Architecturesd
I've used [Nvidia] (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) model as starting point but after many tries, I've changed the model to get better result on my traing and validation data.

Hyperparameters:
* Learning rate: 0.001. I tried different learning rates and found 0.001 to be the most effective learning rate.
* Activation function: ELU
* Dropout keep prob: 0.5
* Optimizer: Adam. I chose the [Adam optimizer](https://keras.io/optimizers/#adam), because it's extremly robust and working well.
* Loss Function: MSE

The network consists of 9 layers, including a normalization layer, 4 convolutional layers and 4 fully connected layers along with a 50% dropout layer after each fully connected layer. The total number of parameters is 400,123.

I used [Keras' Lamda layer](https://keras.io/layers/core/#lambda) as input layer to normalize all input features on the GPU since it could do it much faster than on a CPU.

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

