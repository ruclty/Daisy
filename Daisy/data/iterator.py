from .dataset import Dataset
import torch
import numpy as np

class Iterator:
	def __init__(self,
                 dataset,
                 batch_size,
                 sort_key = None,
                 reverse = False,
                 shuffle = False,
                 transpose = False,
                 square = False,
                 pad = None,
                 labels = None):
        
		assert batch_size > 1        
		self.batch_size = batch_size
		self.transpose = transpose
		self.labels = labels
		self.dataset = dataset
		self.shuffle = shuffle
		self.concat()
		
		if sort_key is not None:
			idx = self.dataset.columns.index(sort_key)	
			data = sorted(self.data, key=lambda x: x[idx], reverse=reverse)		
			self.data = np.asarray(data)
		
		if self.shuffle:
			state = np.random.get_state()
			np.random.shuffle(self.data)
			if self.label is not None:
				np.random.set_state(state)
				np.random.shuffle(self.label)
		
		if square:
			assert self.labels is None
			assert pad is not None
			shape = int(np.ceil(np.sqrt(self.data.shape[1])))
			pad_num = shape*shape - self.data.shape[1]
			if pad_num != 0:
				padding = pad * np.ones([self.data.shape[0], pad_num])
				self.data = np.concatenate([self.data, padding], axis = 1)
			self.data = self.data.reshape(-1, shape, shape)
			self.shape = shape
		
		idx = 0
		self.batch_x = []
		self.batch_y = []
		while(idx < len(self.data)):
			if idx + batch_size < len(self.data):
				self.batch_x.append(self.data[idx:idx+batch_size])
				if labels is not None:
					self.batch_y.append(self.label[idx:idx+batch_size])
			else:
				self.batch_x.append(self.data[idx:len(self.data)])
				if labels is not None:
					self.batch_y.append(self.label[idx:len(self.data)])
			idx += batch_size
			
		self.batch_x = np.asarray(self.batch_x)
		self.batch_y = np.asarray(self.batch_y)
		self.iter_idx = 0
		
			
	def concat(self):
		columns = []
		labelcols = []
		for col in self.dataset.columns:
			if self.labels is not None and col in self.labels:
				labelcols.append(self.dataset.__dict__[col].convert())
			else:
				columns.append(self.dataset.__dict__[col].convert())
		data = columns[0]
		for i in range(1, len(columns)):
			data = np.concatenate([data, columns[i]], axis = 1)
		self.data = data
		
		self.label = None
		if self.labels is not None:
			label = labelcols[0]
			for i in range(1, len(labelcols)):
				label = np.concatenate([label, labelcols[i]], axis = 1)
			self.label = label
		
	def sample(self, num_sample = None, label = None):
		sample_label = None
		if num_sample is None:
			num_sample = self.batch_size
		if label is not None:
			assert self.label is not None
			assert label in self.label
			index = np.argwhere(self.label == label)[:,0]
			row_rand = np.arange(index.shape[0])
			np.random.shuffle(row_rand)
			num_sample = num_sample if num_sample <= len(row_rand) else len(row_rand)
			row_rand = index[row_rand[0:num_sample]]
			sample_data = self.data[row_rand]
			sample_label = self.label[row_rand]
			
		if label is None:
			row_rand = np.arange(self.data.shape[0])
			np.random.shuffle(row_rand)
			num_sample = num_sample if num_sample <= len(row_rand) else len(row_rand)
			row_rand = row_rand[0:num_sample]
			sample_data = self.data[row_rand]
			sample_label = self.label[row_rand] if self.label is not None else None
		
		x = torch.from_numpy(sample_data.astype(np.float32))
		y = torch.from_numpy(sample_label.astype(np.float32)) if sample_label is not None else None
		
		if y is None:
			return x
		
		return x, y
			
		
	@classmethod
	def split(cls,
              batch_size,
              train = None,
              validation = None,
              test = None,
              labels = None,
              shuffle = False,
              sort_key = None,
              reverse = False,
              transpose = False,
              square = False,
              pad = None,
              ):
		iterator_args = {'batch_size': batch_size,
                         'shuffle': shuffle,
                         'sort_key': sort_key,
                         'reverse': reverse,
                         'transpose': transpose,
                         'square': square,
                         'pad': pad}
		train_it = None if train is None else cls(dataset = train, labels = labels, **iterator_args)
		valid_it = None if validation is None else cls(dataset = validation, labels = labels, **iterator_args)
		test_it = None if test is None else cls(dataset = test, labels = labels, **iterator_args)
		
		iterators = tuple(it for it in (train_it, valid_it, test_it) if it is not None)
		return iterators
	
	def __len__(self):
		return len(self.batch_x)
	
	def __iter__(self):
		self.iter_idx = 0
		return self
	
	def __next__(self):
		if self.iter_idx == len(self.batch_x):
			raise StopIteration
		else:
			x = None
			y = None
			x = self.batch_x[self.iter_idx]
			if self.label is not None:
				y = self.batch_y[self.iter_idx]
			if self.transpose:
				x = x.T
				y = y.T if y is not None else None
			x = torch.from_numpy(x.astype(np.float32))
			if y is not None:
				y = torch.from_numpy(y.astype(np.float32))
			self.iter_idx += 1
		if y is None:
			return x
		return x, y
		
	
		
	
	
