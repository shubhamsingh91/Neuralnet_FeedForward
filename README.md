# Neuralnet_FeedForward

This function compares a feedforward function written in pytorch with the standard feedforward (in Class NN) function. 
Three linear layers of size 128, 64 and 10 are used along with ReLU and Softmax activation functions.
A test data (of size 64x1x28x28) from MNIST is chosen for the test.
The output is the norm of the difference in the model output vs the feedforward function output.
The main purpose of this test is to get a deeper understanding in the Feed Forward step of the Multi-layer Perceptron Training method.
