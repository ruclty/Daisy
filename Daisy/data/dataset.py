import pandas as pd
import numpy as np
import os, sys
import copy
from .field import NumericalField, CategoricalField

class Dataset:
	def __init__(self, 
				fields,
				path, 
				ignore_columns=None,
				format='csv',
				ratio=1):
				
		self.fields = fields
		if format == "csv":
			self.raw_file = pd.read_csv(path)
			if ratio != 1:
				self.raw_file = self.raw_file.sample(frac=ratio, replace=False)
			self.columns = list(self.raw_file)
			if ignore_columns is not None:
				self.columns = list(filter(lambda x: x not in ignore_columns, self.columns))
				
		self.check_header()
		
		#Read data by row
		self.Rows = []
		for idx in range(len(self.raw_file)):
			values = []
			for col in self.columns:
				values.append(self.raw_file.iloc[idx][col])	
			self.Rows.append(Row(values, idx, self.columns))	
		#Read data by column
		self.dim = 0
		self.col_type = {}
		self.col_dim = []
		self.col_ind = []
		for col, field in fields:
			if col not in self.columns:
				continue
			assert field is None or isinstance(field, (NumericalField, CategoricalField))
			if isinstance(field, CategoricalField):
				values = self.raw_file[col].astype(str).values.reshape(-1,1)
			else:
				values = self.raw_file[col].values.reshape(-1, 1)
			Field = field
			self.__setattr__(col, Column(values, name=col, field=Field) )
			self.col_type[col] = "numerical" if isinstance(Field, NumericalField) else "categorical"
			
								
	def check_header(self):
		for col, field in self.fields:
			if col not in self.columns:
				sys.exit("{} Not Exist !".format(col))
	
	def learn_convert(self):
		self.dim = 0
		sta = 0
		end = 0
		for col, field in self.fields:
			if field is not None:
				field.learn()
			self.__dict__[col].convert_data()
			self.col_dim.append(self.__dict__[col].dim)
			self.dim += self.__dict__[col].dim
			sta = end
			end = self.dim
			self.col_ind.append((sta, end))
	
	@classmethod
	def split(cls,
              fields,
              path,
              train = None, 
              validation = None,
              ignore_columns = None,
              format='csv',
              test = None,
              train_ratio=1,
              test_ratio=1,
              valid_ratio=1):
			
		dataset_args = {'fields': fields, 'ignore_columns': ignore_columns, 'format': format}
		train_data = None if train is None else cls(
			path = os.path.join(path, train), **dataset_args, ratio=train_ratio)
		valid_data = None if validation is None else cls(
			path = os.path.join(path, validation), **dataset_args, ratio=valid_ratio)
		test_data = None if test is None else cls(
			path = os.path.join(path, test), **dataset_args, ratio=test_ratio)
		datasets = tuple(
			d for d in (train_data, valid_data, test_data) if d is not None)
			
		return datasets
		
	def reverse(self, sample):
		assert isinstance(sample, np.ndarray)
		assert sample.shape[1] == self.dim
		sample_columns = []
		sta = 0
		end = 0
		for col in self.columns:
			dim = self.__dict__[col].dim
			sta = end
			end += dim
			sample_columns.append(sample[:,sta:end])
		
		for i, col in enumerate(self.columns):
			sample_columns[i] = self.__dict__[col].reverse(sample_columns[i])
		
		sample_table = sample_columns[0]
		for i in range(1, len(sample_columns)):
			sample_table = np.concatenate([sample_table, sample_columns[i]], axis=1)
		
		return sample_table
			
	
	@staticmethod
	def load_cache():
		return 0
	
	def save_cache(self):
		return 0
		
	def __getitem__(self, i):
		return self.Rows[i]
	
	def __len__(self):
		return len(self.Rows)
		
	def head(self):
		return self.raw_file.head()
	
	def tail(self):	
		return self.raw_file.head()		

class Column:
	def __init__(self, values, name, field):
		self.values = values
		self.name = name
		self.Field = field
		if self.Field is not None:
			self.Field.get_data(values)
		
		
	def __str__(self):
		return repr(self.values)
		
	__repr__ = __str__
	
	def __getitem__(self, i):
		return self.values[i]
	
	def convert_data(self):
		if self.Field is None:
			self.conv_values = self.values
			self.dim = 1
		else:
			self.conv_values = self.Field.convert(self.values)
			self.dim = self.Field.get_dim()
	
	def convert(self):
		return self.conv_values
	
	def reverse(self, features):
		if self.Field is not None:
			data = self.Field.reverse(features)
			data = data.astype(self.values.dtype)
			return data
		return features

class Row:
	def __init__(self, values, idx, columns):
		assert len(values) == len(columns)
		self.columns = columns
		self.idx = idx
		self.values = values
		for i in range(len(values)):
			self.__setattr__(columns[i], values[i])
		
	def __str__(self):
		return repr(self.values)
	
	__repr__ = __str__
	
	
	
	
	
	
				 
