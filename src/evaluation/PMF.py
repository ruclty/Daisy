# -*- coding: utf-8 -*-
import numpy as np
import math

class PMF(object):
    
    def __init__(self, data, continuous = False, interval = 1):
        self.data = data
        self.count = len(self.data)
        self.continuous = continuous
        self.interval = interval
        if self.continuous == True:
            self.minimum = math.floor(min(self.data) - min(self.data) % self.interval)
            self.maximum = math.ceil(max(self.data) - max(self.data) % self.interval + self.interval)
        self.compute_PMF()


    def compute_PMF(self):
        self.dict = {}
        if self.continuous == False:
            for item in self.data:
                self.dict[item] = self.dict.get(item, 0) + 1
            self.dict = dict(sorted(self.dict.items(), key=lambda x:x[0]))
            self.keys = list(self.dict.keys())
        else:
            i = self.minimum
            while i < self.maximum:
                self.dict[i] = 0
                i = round(i + self.interval, 3)
            self.keys = list(self.dict.keys())
            for item in self.data:
                index = self.calculate_key(item)
                self.dict[index] = self.dict.get(index, 0) + 1
            self.dict = dict(sorted(self.dict.items(), key=lambda x:x[0]))
        self.values = list(self.dict.values())
        self.probability = [round(i/self.count, 3) for i in self.values]
        self.PMF = np.column_stack((self.keys, self.probability))


    def interval_align(self, minimum=None, maximum=None, keys=None):
        if self.continuous == True:
            assert minimum is not None
            assert maximum is not None
            self.minimum = minimum
            self.maximum = maximum
            i = self.minimum
            while i < self.maximum:
                if i not in self.dict:
                    self.dict[i] = 0
                i = round(i + self.interval, 3)
            self.dict = dict(sorted(self.dict.items(), key=lambda x:x[0]))
            self.values = list(self.dict.values())
            self.keys = list(self.dict.keys())
            self.probability = [round(i/self.count, 3) for i in self.values]
            self.PMF = np.column_stack((self.keys, self.probability))
        else:
            assert keys is not None
            for key in keys:
                if key not in self.dict:
                    self.dict[key] = 0
            self.dict = dict(sorted(self.dict.items(), key=lambda x:x[0]))
            self.values = list(self.dict.values())
            self.keys = list(self.dict.keys())
            self.probability = [round(i/self.count, 3) for i in self.values]
            self.PMF = np.column_stack((self.keys, self.probability))
    

    def calculate_key(self, value):
        if self.continuous == False:
            return value
        else:
            if value < self.keys[0]:
                return self.keys[0]
            for i in range(len(self.keys) - 1):
                if self.keys[i] <= value and self.keys[i+1] > value:
                    return self.keys[i]
            return self.keys[-1]