#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:01:28 2018

@author: whitneybai
"""

import time

class Timer(object):
    def __init__(self):
        self.start_time = 0.
        self.tot_time = 0.
        #self.tot = 0.
        #self.calls = 0.
        #self.aver_time = 0.
        
    def beginr(self):
        self.start_time = time.time()
        
    def total(self):
        self.tot_time = time.time() - self.start_time
        
        return self.tot_time