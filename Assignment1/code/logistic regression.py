#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 22:06:13 2018

@author: BAI Haoyue

Logistic regression model with gradient descent optimization algorithm
"""

import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(1)


# Datapath
datasetPath = './datasets/mouse.npz'
#datasetPath = './datasets/pulsar_star.npz'
#datasetPath = './datasets/wine.npz'


# Parameters
initalpha = 0.001
maxIterations = 400000
numTests = 5
train_accuracy = 0
test_accuracy = 0
list_trainAcc = []
list_testAcc = []
list_loss = []
list_iteration = []


# Functions
def dataSetLoader(data_path):
    dataSet = np.load(data_path)
    train_data = np.mat(dataSet['train_X'])
    train_label = np.mat(dataSet['train_Y'])
    test_data = np.mat(dataSet['test_X'])
    test_label = np.mat(dataSet['test_Y'])
    return train_data, train_label, test_data, test_label

def sigmoid(X):
    return 1.0/(1+np.exp(-X))

def layerSize(input_data, input_label):
    numSamples, numFeatures = np.shape(input_data)
    numLabels,numlabelSamples = np.shape(input_label)
    assert (numSamples == numlabelSamples)
    return numSamples, numFeatures

def initWeights(nn_X, nn_Y=1):
    weights = np.random.rand(nn_X, nn_Y) * 0.02 - 0.01
    return weights

def forwardPropagation(input_data, weights):
    hypothesis = sigmoid(np.dot(input_data, weights))
    return hypothesis, weights

def computeLoss(hypothesis, input_label):
    N = input_label.shape[1]
    logperloss = np.multiply(np.log(hypothesis), input_label.T) + np.multiply(np.log(1-hypothesis), 1-input_label.T)
    crossEntropyLoss = -1/N * np.sum(logperloss)
    return crossEntropyLoss    

def backPropagation(input_data, hypothesis, input_label):
    N = input_label.shape[1]
    dO = hypothesis - input_label.T
    input_label = np.mat(input_label)
    gradient = 1.0/N * np.dot(input_data.T, dO)
    return gradient

def updataWeights(weights, gradient, initalpha, i):
    #alpha = initalpha
    alpha = 1.0/(1 + i) + initalpha
    weights = weights - alpha * gradient
    return weights

def predict(weights, input_data):
    hypothesis, weights = forwardPropagation(input_data, weights)
    predictions = (hypothesis > 0.5)    
    return predictions

# Build logistic regression model and adopt gradient descent optimization algorithm
def nnModel(input_data, input_label, test_data, test_label, maxIterations, initalpha, print_acc = False):
    numSamples, numFeatures = layerSize(input_data, input_label)
    weights = initWeights(numFeatures)
    for i in range(0, maxIterations):
        hypothesis, weights = forwardPropagation(input_data, weights)
        crossEntropyLoss = computeLoss(hypothesis, input_label)
        gradient = backPropagation(input_data, hypothesis, input_label)
        weights = updataWeights(weights, gradient, initalpha, i)       
        if print_acc and i % 1000 == 0:
            list_iteration.append(i/1000)
            predictions = predict(weights, input_data)
            accuracy = float((np.dot(input_label, predictions) + np.dot(1-input_label, 1-predictions))/float(input_label.size)*100)
            list_trainAcc.append(accuracy/100)
            crossEntropyLoss = computeLoss(hypothesis, input_label)
            list_loss.append(crossEntropyLoss)
            test_predictions = predict(weights, test_data)
            test_accuracy = float((np.dot(test_label, test_predictions) + np.dot(1-test_label, 1-test_predictions))/float(test_label.size)*100)
            list_testAcc.append(test_accuracy/100)
            print("trainAccuracy after iteration %i: % f" %(i, accuracy))
            print("testAccuracy after iteration %i: % f" %(i, test_accuracy))
            print("traincrossEntropyLoss after iteration %i: %f"%(i, crossEntropyLoss))
    return weights
    
# Load dataset
begin_time = time.time() # compute time
train_data, train_label, test_data, test_label = dataSetLoader(datasetPath)

# Preprocess
dimTrain = train_data.shape[0]
X0 = np.tile(1.0, (dimTrain, 1))
train_data = np.mat(np.concatenate((X0, train_data), axis=1))
dimTest = test_data.shape[0]
Y0 = np.tile(1.0, (dimTest, 1))
test_data = np.mat(np.concatenate((Y0, test_data), axis=1))

# Training
weights = nnModel(train_data, train_label, test_data, test_label, maxIterations, initalpha, print_acc=True)

# Testing
for j in range(numTests):
    test_predictions = predict(weights, test_data)
    test_accuracy += float((np.dot(test_label, test_predictions) + np.dot(1-test_label, 1-test_predictions))/float(test_label.size)*100)
    train_predictions = predict(weights, train_data)
    train_accuracy += float((np.dot(train_label, train_predictions) + np.dot(1-train_label, 1-train_predictions))/float(train_label.size)*100)
total_time = time.time() - begin_time
print("after %d repeat the average error rate on training sets is:%f" %(numTests, train_accuracy/float(numTests)))
print("after %d repeat the average error rate on test sets is:%f" %(numTests, test_accuracy/float(numTests)))
print("Total test time is: %.4fs" % (total_time))

# Record and visualize the change in performance during training
plt.figure()
plt.plot(list_iteration, list_loss, color='yellow', linewidth=2.0, linestyle='-', label='train loss')
plt.plot(list_iteration, list_trainAcc, color='red', linewidth=2.0, linestyle='-', label='train accuracy')
plt.plot(list_iteration, list_testAcc, color='green', linewidth=2.0, linestyle='-', label='test accuracy')
plt.legend(loc='lower left')
plt.xlim((0, 400))
plt.ylim((0, 1.0))
plt.xlabel('iterations(*10^3)')
plt.ylabel('Performance during training')
plt.title('Dataset: Pulsar star')
plt.savefig("PulsarStar.png")
plt.show()













