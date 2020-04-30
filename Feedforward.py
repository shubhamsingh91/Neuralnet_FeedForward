## Testing forward function on MNIST data

# Modified on 4/26/20

# Loading an image dataset

# These are 32 bit float tensors here

import numpy as np
import torch

from torch import nn
import torch.nn.functional as F
import torch.optim as optim 

import helper

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# transform = transforms.Compose([transforms.ToTensor(),
# transforms.Normalize((0.5,), (0.5,))
# ])

# # Download and load the test set
# testset = datasets.MNIST('MNIST_data/',download=True, train=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = True)

# dataiter = iter(testloader)
# images, labels = dataiter.next()
# images.resize_(images.shape[0],784) # Resizing the image to 64x784 from 64x1x28x28

#--- Loading the image dataset here

images = torch.load('images.pt') # 64x784


##---------- All functions below this-------------------------##

# defining the ReLU function here

# input x 
def relu_fun(x):
    if (x<0):
        out = 0
    elif (x>=0):
        out = x
    return out

# input x matrix
def relu_fun_mat(x):
    n = x.size(0)
    m = x.size(1)
    for ii in range(n):
        for jj in range(m):
            x[ii][jj] = relu_fun(x[ii][jj])
            
    return x

def softmax_fun(x):
    m = x.size(1) # number of columns in x
    n = x.size(0) # number of rows in x
    out = torch.zeros([n,m], dtype=torch.float32) # Declaring the n tensor
    
    for jj in range(m): # for all columns
        out[:,jj] = np.exp(x[:,jj])/float(sum(np.exp(x[:,jj]))) 
    
    return out


# Input n is the size of bias
def out_layer(x,weight,bias,n):
    n = bias.size(0)
    x = torch.mm(weight, x)
    x = tens_to_arr(x)
    bias = np.reshape(bias,(n,1))
    x = x + bias

# Convert tensor to array
def tens_to_arr(x):
    out = np.asarray(x)
    return out

# Function for implementing a linear layer in feed-forward using pre-defined weights and bias
# input is x matrix, weight and bias matrices
# returns x tensor with linear layer implemented on it

def feed_fw_linear(x,weight,bias):
    n_bias = bias.shape[0]
    x = torch.mm(weight, x)# + bias1
    x = tens_to_arr(x)
    bias1 = tens_to_arr(bias)
    bias = np.reshape(bias,(n_bias,1))
    x = x + bias
    x = torch.Tensor(x)
    return x

##---------- Feed-forward implementation here-----------------## 

# Loading weights and bias matrix here
weight1 = torch.load('weight1.pt') # size 128x784
bias1 = torch.load('bias1.pt') # size 128x1

weight2 = torch.load('weight2.pt') # size 64x128
bias2 = torch.load('bias2.pt') # size 64

weight3 = torch.load('weight3.pt') # size 10x64
bias3 = torch.load('bias3.pt') # size 10x1

x = images # size 64x784
x = x.t() # transposed it to 784x64

# Feed Forward Model
# Linear layer 1- 784 to 128
x = feed_fw_linear(x,weight1,bias1)

# # ReLU activation function - 1 to 1
x = relu_fun_mat(x)

# # Linear layer 2- 128 to 64
x = feed_fw_linear(x,weight2,bias2)

# # ReLU activation function
x = relu_fun_mat(x)

# # Linear layer 3- 64 to 10
x = feed_fw_linear(x,weight3,bias3)

# # Softmax layer, applying softmax to 10x64
x = softmax_fun(x)

#finally transposing it back to 64x10
x = x.t()

# testing from forward model(from nn class forward function)
y = torch.load('output.pt')

z = x - y
u= torch.norm(z,p=2)
print('Norm of difference is', u)


