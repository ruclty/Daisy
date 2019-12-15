import numpy as np
import math
from sklearn.mixture import GaussianMixture

class NumericalField:
# Field for numerical column
	dtype = "Numerical Data"
	def __init__(self, model="gmm", n=5):
		self.model_name = model
		if model == "gmm":
			self.n = n
			self.model = GaussianMixture(n)
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
			
		elif self.model_name == "normalize":
			self.Min = np.min(self.fitdata)
			self.Max = np.max(self.fitdata)
			self.K = 2/(self.Max-self.Min)
				
		self.learned = True		
	
	def get_data(self, data):
		assert isinstance(data, np.ndarray)
		assert data.shape[1] == 1
		if self.fitdata is None:
			self.fitdata = data
		else:
			self.fitdata = np.concatenate([self.fitdata, data], axis=0)
	
	def convert(self, data):
		assert isinstance(data, np.ndarray)
		assert data.shape[1] == 1
		if self.model_name == "gmm":
			features = (data - self.means) / (2 * self.stds)
			probs = self.model.predict_proba(data)
			argmax = np.argmax(probs, axis=1)	
			idx = np.arange(len(features))
			features = features[idx, argmax].reshape(-1, 1)
			features = np.concatenate((features, probs), axis=1)
		elif self.model_name == "normalize":
			features = -1.0 + self.K * (data - self.Min)
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
		elif self.model_name == "normalize":
			assert features.shape[1] == 1
			data = (features + 1)/self.K + self.Min
		return data
	
	def get_dim(self):
		if self.model_name == "gmm":
			return self.n + 1
		elif self.model_name == "normalize":
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
		self.vset = np.unique(self.fitdata)
		for idx, v in enumerate(self.vset):
			self.dict[v] = idx
			self.rev_dict[idx] = v
		if self.method == "dict":
			self.Min = 0
			self.Max = len(self.vset)-1
			self.K = 1/(self.Max-self.Min)
	
	def get_data(self, data):
		if self.fitdata is None:
			self.fitdata = data
		else:
			self.fitdata = np.concatenate([self.fitdata, data], axis=0)
			
	def convert(self, data):
		assert isinstance(data, np.ndarray)
		assert data.shape[1] == 1
        
		origin = data
		data = data.reshape(1, -1)
		data = list(map(lambda x: self.dict[x], data[0]))
		data = np.asarray(data).reshape(-1, 1)
		features = data
		if self.method == "dict":
			self.dim = 1
			features = self.K * (data - self.Min)
			return features
			
		if self.method == "binary":
			#print(self.vset)
			if len(self.dict) <= 2:
				self.dim = 1
				return features
			self.dim = int(math.log(len(self.dict)-1, 2)) + 1
			features = np.zeros((data.shape[0], self.dim), dtype="int")
			for i in range(data.shape[0]):
				for j in range(self.dim):
					if 2**j & data[i][0] != 0:
						features[i][self.dim-1-j] = 1
			return features
			
		if self.method == "one-hot":
			features = np.zeros((data.shape[0], len(self.dict)), dtype="int")
			idx = np.arange(len(features))
			features[idx, data.reshape(1, -1)] = 1
			self.dim = len(self.dict)
			if self.noise is not None:
				noise = np.random.uniform(0,self.noise,features.shape)
				features = features + noise
				temp = np.sum(features, axis=1).reshape(-1,1)
				features = features/temp
		
		return features
		
	def reverse(self, features):
		assert isinstance(features, np.ndarray)
		assert features.shape[1] == self.dim
		if self.method == "binary":
			data = features > 0.5
			data = data.astype(int)
			features = np.zeros((data.shape[0], 1),dtype="int")
			for i in range(data.shape[0]):
				for j in range(self.dim):
					features[i][0] += data[i][j] * (2**(self.dim-1-j))
			
		if self.method == "one-hot":
			features = np.argmax(features, axis=1)
		
		if self.method == "dict":
			features = features/self.K + self.Min
			features = features + 0.5
			features = features.astype(int)
			
		features = features.reshape(1, -1)
		data = list(map(lambda x: self.rev_dict[x], features[0]))
		data = np.asarray(data).reshape(-1, 1)
		
		return data	
	
	def get_dim(self):
		return self.dim
	
	
	
