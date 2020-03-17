#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 17:03:52 2018

@author: BAI Haoyue

Build the SVM with linear kernel and SVM with RBF kernel
Tune the parameters with cross validation techniques
"""

import numpy as np
import time
from sklearn import svm

# Datapath
# datasetPath = './datasets/mouse.npz'
# datasetPath = './datasets/pulsar_star.npz'
datasetPath = './datasets/wine.npz'

# Functions
def dataSetLoader(data_path):
    dataSet = np.load(data_path)
    train_data = dataSet['train_X']
    train_label = dataSet['train_Y']
    test_data = dataSet['test_X']
    test_label = dataSet['test_Y']
    return train_data, train_label, test_data, test_label

# Load data
begin_time = time.time() # compute time
train_data, train_label, test_data, test_label = dataSetLoader(datasetPath)

# Training
svmLinear = svm.SVC(kernel='linear')
svmLinear.fit(train_data, train_label)

# Test
train_accuracy = svmLinear.score(train_data, train_label)
test_accuracy = svmLinear.score(test_data, test_label)
total_time = time.time() - begin_time
print("for SVM with linear kernel the accuracy on training sets is:%f" %(train_accuracy))
print("for SVM with linear kernel the accuracy on test sets is:%f" %(test_accuracy))
print("Total train and test time is: %.4fs" % (total_time))

