#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 17:24:31 2018

@author: BAI Haoyue

Build the SVM with linear kernel and SVM with RBF kernel
Tune the parameters with cross validation techniques
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

# Datapath
datasetPath = './datasets/mouse.npz'
# datasetPath = './datasets/pulsar_star.npz'
# datasetPath = './datasets/wine.npz'

# Parameters
valueC = [1, 1.0/10, 1.0/100, 1.0/1000]
numCrossVal = 5
list_x = [0, 1, 2, 3]
list_valAcc = []
best_c = 0
best_valAcc = 0

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

# Cross validation to choose best hyperparameters
for i, c in enumerate(valueC):
    val_accuracy = 0
    for j in range(numCrossVal):
        cross_train_data, cross_val_data, cross_train_label, cross_val_label = \
            train_test_split(train_data, train_label.T, test_size=0.2, random_state=0)
        cross_train_label = cross_train_label.T
        cross_val_label = cross_val_label.T
        svmRBF = svm.SVC(C=c, kernel='rbf')
        svmRBF.fit(cross_train_data, cross_train_label)
        val_accuracy += svmRBF.score(cross_val_data, cross_val_label)
    list_valAcc.append(val_accuracy/float(numCrossVal))
    if val_accuracy/float(numCrossVal) >= best_valAcc:
        best_valAcc = val_accuracy/float(numCrossVal)
        best_c = c
    print ("Accuracy for value C {} is:{} %".format(c, 100 * val_accuracy/float(numCrossVal)))
print ("Accuracy for best value C {} is:{} %".format(best_c, 100 * best_valAcc))

# Training
final_svmRBF = svm.SVC(C=best_c, kernel='linear')
final_svmRBF.fit(train_data, train_label)

# Testing
train_accuracy = final_svmRBF.score(train_data, train_label)
test_accuracy = final_svmRBF.score(test_data, test_label)
total_time = time.time() - begin_time
print("Dataset: wine")
print("for SVM with linear kernel the accuracy on training sets is:%f" %(train_accuracy))
print("for SVM with linear kernel the accuracy on test sets is:%f" %(test_accuracy))
print("Total train and test time is: %.4fs" % (total_time))

# Visualize the parameter tuning result of SVM with RBF using cross validation
plt.figure()
plt.plot(list_x, list_valAcc, color='blue', linewidth=2.0, linestyle='-',marker='*', label='average_valAcc')
plt.legend(loc='lower right')
plt.xlim((0, 3))
plt.ylim((0, 1.0))
plt.xlabel('value candidates (10 ^ (-x))')
plt.ylabel('Average of val accuracy using cross validation')
plt.title('Dataset: Mouse')
plt.savefig("4_Mouse.png")
plt.show()




