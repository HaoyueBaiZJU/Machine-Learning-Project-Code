#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 10:58:43 2018

@author: BAI Haoyue

Neural network model with gradient descent optimization algorithm
"""


import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


np.random.seed(1)


# Datapath
datasetPath = './datasets/mouse.npz'
#datasetPath = './datasets/pulsar_star.npz'
#datasetPath = './datasets/wine.npz'

# Parameters
maxIterations = 20000
alpha = 0.01
numHiddenLayers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
numCrossVal = 5
val_accuracy = 0
best_numH = 0
best_valAcc = 0
list_valAcc = []


# Functions
def dataSetLoader(data_path):
    dataSet = np.load(data_path)
    train_data = dataSet['train_X']
    train_label = dataSet['train_Y']
    test_data = dataSet['test_X']
    test_label = dataSet['test_Y']
    return train_data, train_label, test_data, test_label

def layerSize(inputData, inputLabels):
    a, b = np.shape(inputData)
    return b, 1

def initParameters(nn_X, nn_hidden, nn_Y):
    np.random.seed(2)
    W1 = np.random.randn(nn_hidden, nn_X) * 0.01
    b1 = np.zeros((nn_hidden, 1))
    W2 = np.random.randn(nn_Y, nn_hidden) * 0.01
    b2 = np.zeros((nn_Y, 1))    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def forwardPropagation(inputData, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, inputData.T) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)    
    weights = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, weights

def computeCost(A2, inputLabels, parameters):
    m = inputLabels.shape[0]
    logprobs = np.multiply(np.log(A2), inputLabels) + np.multiply(np.log(1-A2), 1-inputLabels)
    cost = -1/m * np.sum(logprobs)
    cost = np.squeeze(cost)
    assert(isinstance(cost, float))
    
    return cost

def backPropagation(parameters, weights, inputData, inputLabels):
    m = inputData.shape[1]
    
    W2 = parameters["W2"]
    
    A1 = weights["A1"]
    A2 = weights["A2"]
    
    dZ2 = A2 - inputLabels
    dW2 = 1.0/m * np.dot(dZ2, A1.T)
    db2 = 1.0/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1.0/m * np.dot(dZ1, inputData)
    db1 = 1.0/m * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def updateParameters(parameters, grads, alpha):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(inputData, inputLabels, nn_hidden, num_iterations, alpha, print_cost = False):
    np.random.seed(3)
    nn_X, nn_Y = layerSize(inputData, inputLabels)
    parameters = initParameters(nn_X, nn_hidden, nn_Y)
    
    for i in range(0, num_iterations):
        A2, cache = forwardPropagation(inputData, parameters)
        cost = computeCost(A2, inputLabels, parameters)
        grads = backPropagation(parameters, cache, inputData, inputLabels)
        parameters = updateParameters(parameters, grads, alpha)
        
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
    
    return parameters

def predict(parameters, inputData):
    A2, weights = forwardPropagation(inputData, parameters)
    predictions = (A2 > 0.5)
    
    return predictions

# Load dataset
begin_time = time.time() # compute time
train_data, train_label, test_data, test_label = dataSetLoader(datasetPath)


# Cross validation to choose best hyperparameters
for i, nn_hidden in enumerate(numHiddenLayers):
    val_accuracy = 0
    for j in range(numCrossVal):
        cross_train_data, cross_val_data, cross_train_label, cross_val_label = \
            train_test_split(train_data, train_label.T, test_size=0.2, random_state=0)
        cross_train_label = cross_train_label.T
        cross_val_label = cross_val_label.T
        parameters = nn_model(cross_train_data, cross_train_label, nn_hidden, maxIterations, alpha)
        predictions = predict(parameters, cross_val_data)
        val_accuracy += float((np.dot(cross_val_label, predictions.T) + np.dot(1-cross_val_label, 1-predictions.T))/float(cross_val_label.size)*100)
    list_valAcc.append(val_accuracy/(float(numCrossVal) * 100))
    if val_accuracy/float(numCrossVal) > best_valAcc:
        best_valAcc = val_accuracy/float(numCrossVal)
        best_numH = nn_hidden
    print ("Accuracy for {} hidden units:{} %".format(nn_hidden, val_accuracy/float(numCrossVal)))
print ("Accuracy for best hidden units number {} is:{} %".format(best_numH, best_valAcc))



# Training
parameters = nn_model(train_data, train_label, best_numH, maxIterations, alpha)
train_predictions = predict(parameters, train_data)
train_accuracy = float((np.dot(train_label, train_predictions.T) + np.dot(1-train_label, 1-train_predictions.T))/float(train_label.size)*100)

# Testing
test_predictions = predict(parameters, test_data)
test_accuracy = float((np.dot(test_label, test_predictions.T) + np.dot(1-test_label, 1-test_predictions.T))/float(test_label.size)*100)
total_time = time.time() - begin_time
print("for best hidden unites number %d the accuracy on training sets is:%f" %(best_numH, train_accuracy))
print("for best hidden unites number %d the accuracy on test sets is:%f" %(best_numH, test_accuracy))
print("Total train and test time is: %.4fs" % (total_time))

# Visualize the parameter tuning result of NN using cross validation
plt.figure()
plt.plot(numHiddenLayers, list_valAcc, color='blue', linewidth=2.0, linestyle='-',marker='*', label='average_valAcc')
plt.legend(loc='lower right')
plt.xlim((1, 10))
plt.ylim((0, 1.0))
plt.xlabel('Number of hidden units')
plt.ylabel('Average of val accuracy using cross validation')
plt.title('Dataset: Mouse')
plt.savefig("2_Mouse.png")
plt.show()


