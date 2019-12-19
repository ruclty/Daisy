# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 02:51:38 2019

@author: susie
"""

import numpy as np
from sklearn.mixture import GaussianMixture

class NumericalField:
    # Field for numerical column
    dtype = "Numerical Data"
    def __init__(self, model="gmm", n=5):
        self.model_name = model
        if model == "gmm":
            self.n = n
            self.model = GaussianMixture(n)
        elif model == "minmax":
            pass
        self.fitdata = None
        self.learned = False
			
    def learn(self):
        if self.learned:
            return
        if self.model_name == "gmm":
            self.model.fit(self.fitdata)
			#get the mixture distribution parameters: weights, means, stds
            self.weights = self.model.weights_
            self.means = self.model.means_.reshape((1, self.n))[0]
            self.stds = np.sqrt(self.model.covariances_).reshape((1, self.n))[0]		
        elif self.model_name == "minmax":
            self.Min = np.min(self.fitdata)
            self.Max = np.max(self.fitdata)
            self.K = 2/(self.Max-self.Min)
        elif self.model_name == 'meanstd':
            self.mu = np.mean(self.fitdata)
            self.sigma = np.std(self.fitdata)
        else: 
            print('invalid encoding method')
        self.learned = True		
        
    def get_data(self, data):
        if self.fitdata is None:
            self.fitdata = data
        else:
            self.fitdata = np.concatenate([self.fitdata, data], axis=0)
	
    def convert(self, data):
        assert isinstance(data, np.ndarray)
        if self.model_name == "gmm":
            features = (data - self.means) / (2 * self.stds)
            probs = self.model.predict_proba(data)
            argmax = np.argmax(probs, axis=1)	
            idx = np.arange(len(features))
            features = features[idx, argmax].reshape(-1, 1)
            features = np.concatenate((features, probs), axis=1)
        elif self.model_name == "minmax":
            features = -1.0 + self.K * (data - self.Min)
        elif self.model_name == 'meanstd':
            features = (data - self.mu)/self.sigma
        else: 
            print('invalid encoding method')
        return features
	
    def reverse(self, features):
        assert isinstance(features, np.ndarray)
        if self.model_name == "gmm":
            assert features.shape[1] == self.n + 1
            v = features[:, 0]
            u = features[:, 1:self.n+1].reshape(-1, self.n)
            argmax = np.argmax(u, axis=1)
            mean = self.means[argmax]
            std = self.stds[argmax]
            v_ = v * 2 * std + mean
            data = v_.reshape(-1, 1)
        elif self.model_name == "minmax":
            data = (features + 1)/self.K + self.Min
        elif self.model_name == "meanstd":
            data = features*self.sigma + self.mu
        else: 
            print('invalid encoding method')
        return data
	
    def dim(self):
        if self.model_name == "gmm":
            return self.n + 1
        else:
            return 1


class CategoricalField:
# Field for categorical column
	dtype = "Categorical Data"
	def __init__(self, method="one-hot",noise=0.2):
		self.dict = {}
		self.rev_dict = {}
		self.method = method
		self.fitdata = None
		self.noise = noise
		
	def learn(self):
		vset = np.unique(self.fitdata)
		for idx, v in enumerate(vset):
			self.dict[v] = idx
			self.rev_dict[idx] = v
	
	def get_data(self, data):
		if self.fitdata is None:
			self.fitdata = data
		else:
			self.fitdata = np.concatenate([self.fitdata, data], axis=0)
			
	def convert(self, data):
		assert isinstance(data, np.ndarray)
		data = data.reshape(1, -1)
			
		data = list(map(lambda x: self.dict[x], data[0]))
		data = np.asarray(data).reshape(-1, 1)
		features = data
		if self.method == "dict":
			return features
			
		if self.method == "one-hot":
			features = np.zeros((data.shape[0], len(self.dict)), dtype="int")
			idx = np.arange(len(features))
			features[idx, data.reshape(1, -1)] = 1
			if self.noise is not None:
				noise = np.random.uniform(0,self.noise,features.shape)
				features = features + noise
				temp = np.sum(features, axis=1).reshape(-1,1)
				features = features/temp
				
		return features
		
	def reverse(self, features):
		assert isinstance(features, np.ndarray)
		if self.method == "one-hot":
			assert features.shape[1] == len(self.dict)
			features = np.argmax(features, axis=1)
		rownum = features.shape[0]
		features = features.reshape(1, -1)
		data = list(map(lambda x: self.rev_dict[x], features[0]))
		data = np.asarray(data)
		data = data.reshape((rownum,1))
		return data	
	
	def dim(self):
		return 1 if self.method=="dict" else len(self.dict)	
	
	