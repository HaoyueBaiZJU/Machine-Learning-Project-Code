#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:40:47 2018

@author: whitneybai
"""

import numpy as np

class DataSetLoader():
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = np.load(self.data_path)
        self.train_data = self.data['train_X']
        self.train_label = self.data['train_Y']
        self.test_data = self.data['test_X']
        self.test_label = self.data['test_Y']
        
    def get_data(self):
        return self.train_data, self.train_label, self.test_data, self.test_label
    

        
        
        
        
    