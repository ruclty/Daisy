from .field import CategoricalField, NumericalField
from .dataset import Dataset
from .iterator import Iterator
import pandas as pd
import numpy as np
import os

def get_dataset(path,
				gmm_col,
				dict_col,
				onehot_col,
				train,
				ignore_col=None,
				test=None,
				validation=None,
				format="csv",
				noise=0.2
				):
	train_path = os.path.join(path, train)
	
	fields = []
	if format == "csv":
		train_csv = pd.read_csv(train_path)
		columns = list(train_csv)
		for i, col in enumerate(columns):
			if ignore_col is not None and col in ignore_col:
				continue
			if col in gmm_col:
				fields.append((col, NumericalField("gmm", 5)))
			elif col in dict_col:
				fields.append((col, CategoricalField("dict")))
			elif col in onehot_col:
				fields.append((col, CategoricalField("one-hot", noise)))
			else: fields.append((col, None))

	datasets = Dataset.split(
		fields = fields,
		path = path,
		train = train,
		validation = validation,
		test = test,
		ignore_columns = ignore_col,
		format = format
	)  
	for dt in datasets:
		dt.learn_convert()
	
	return datasets
	

	
	
	
	
	
	
	
