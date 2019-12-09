import itertools
import math
import random
import argparse

import numpy as np
import pandas as pd
import re
from sklearn.datasets import make_circles

from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, Node, BayesianNetwork

def create_gaussian_oracle(dist_type, num_components, n_dim):
	sigmas = np.zeros((n_dim, n_dim))
	s = random.uniform(0.5, 1)
	np.fill_diagonal(sigmas, s)
	
	num_factor = round(math.pow(num_components, 1/n_dim))
	x = num_factor - 1
	factor = range(-x, x+1, 2)
	factors = [factor] * n_dim
	if dist_type == "grid":
		mus = np.array([np.array(mu) for mu in itertools.product(*factors)], dtype=np.float32)
	elif dist_type == "gridr":
		mus = np.array([np.array(mu) + (np.random.rand(2) - 0.5) for mu in itertools.product(*factors)], dtype=np.float32)
	mus = mus[0: num_components]
	return (mus, sigmas)
	
def gaussian_simulator(mus, sigmas, num_samples, path, shuffle = True, ratios = [0.5, 0.5]):
	assert abs(sum(ratios) - 1.0) < 1e-10
	num_components = mus.shape[0]
	n_dim = mus.shape[1]
	samples = np.empty([num_samples, n_dim])
	labels = []
	bsize = int(np.round(num_samples/num_components))
	
	sta = 0
	end = 0
	s = 0
	ranges = []
	for ratio in ratios:
		s += ratio
		sta = end
		end = int(s*num_components)
		ranges.append(range(sta,end))
	print(ranges)
	for i in range(num_components):
		if i in ranges[0]:
			if random.randint(0,9) > 3:
				label = 1
			else:
				label = 0
		if i in ranges[1]:
			if random.randint(0,9) > 3:
				label = 0
			else:
				label = 1
		if (i+1)*bsize >= num_samples:
			samples[i*bsize:num_samples,:] = np.random.multivariate_normal(mus[i], sigmas, size = num_samples-i*bsize)
			if label == 1:
				labels.append(np.ones([num_samples-i*bsize, 1]))
			else:
				labels.append(np.zeros([num_samples-i*bsize, 1]))
		else:
			samples[i*bsize:(i+1)*bsize,:] = np.random.multivariate_normal(mus[i], sigmas, size = bsize)
			if label == 1:
				labels.append(np.ones([bsize, 1]))
			else:
				labels.append(np.zeros([bsize, 1]))
	labels = np.concatenate(labels, axis = 0)
	samples = np.concatenate((samples,labels), axis = 1)
	if shuffle:
		np.random.shuffle(samples)
	df = pd.DataFrame(samples,index=None)
	df.to_csv(path,index=None)
	
def map_col(index2str, values):
    mapper = dict([(k, v) for v, k in enumerate(index2str)])
    return [mapper[item.decode('utf8')] for item in values]

class MultivariateMaker(object):
    """base class for simulated bayesian network"""

    def __init__(self, dist_type):
        self.model = None

    def sample(self, n):
        nodes_parents = self.model.structure
        processing_order = []

        while len(processing_order) != len(nodes_parents):
            update = False

            for id_, parents in enumerate(nodes_parents):
                if id_ in processing_order:
                    continue

                flag = True
                for parent in parents:
                    if not parent in processing_order:
                        flag = False

                if flag:
                    processing_order.append(id_)
                    update = True
            assert update

        data = np.empty((n, len(nodes_parents)), dtype='S128')
        for current in processing_order:
            distribution = self.model.states[current].distribution
            if type(distribution) == DiscreteDistribution:
                data[:, current] = distribution.sample(n)
            else:
                assert type(distribution) == ConditionalProbabilityTable
                parents_map = nodes_parents[current]
                parents = distribution.parents
                for _id in range(n):
                    values = {}
                    for i in range(len(parents_map)):
                        tmp = data[_id, parents_map[i]]
                        try:
                            tmp = tmp.decode('utf8')
                        except:
                            pass
                        values[parents[i]] = tmp
                    data[_id, current] = distribution.sample(parent_values=values)

        data_t = np.zeros(data.shape)
        for col_id in range(data.shape[1]):
            data_t[:, col_id] = map_col(self.meta[col_id]['i2s'], data[:, col_id])
        return data_t




class ChainMaker(MultivariateMaker):

    def __init__(self):
        A = DiscreteDistribution({'1': 1./3, '2': 1./3, '3': 1./3})
        B = ConditionalProbabilityTable(
            [['1','1',0.5],
            ['1','2',0.5],
            ['1','3',0],
            ['2','1',0],
            ['2','2',0.5],
            ['2','3',0.5],
            ['3','1',0.5],
            ['3','2',0],
            ['3','3',0.5],
            ],[A])
        C = ConditionalProbabilityTable(
            [['1','1',0.5],
            ['1','2',0.5],
            ['1','3',0],
            ['2','1',0],
            ['2','2',0.5],
            ['2','3',0.5],
            ['3','1',0.5],
            ['3','2',0],
            ['3','3',0.5],
            ],[B])

        s1 = Node(A, name="A")
        s2 = Node(B, name="B")
        s3 = Node(C, name="C")

        model = BayesianNetwork("ChainSampler")
        model.add_states(s1, s2, s3)
        model.add_edge(s1, s2)
        model.add_edge(s2, s3)
        model.bake()
        self.model = model

        meta = []
        for i in range(self.model.node_count()):
            meta.append({
                "name": chr(ord('A') + i),
                "type": "categorical",
                "size": 3,
                "i2s": ['1', '2', '3']
        })
        self.meta = meta


class TreeMaker(MultivariateMaker):
    def __init__(self):
        A = DiscreteDistribution({'1': 1./3, '2': 1./3, '3': 1./3})
        B = ConditionalProbabilityTable(
            [['1','1',0.5],
            ['1','2',0.5],
            ['1','3',0],
            ['2','1',0],
            ['2','2',0.5],
            ['2','3',0.5],
            ['3','1',0.5],
            ['3','2',0],
            ['3','3',0.5],
            ],[A])
        C = ConditionalProbabilityTable(
            [['1','4',0.5],
            ['1','5',0.5],
            ['1','6',0],
            ['2','4',0],
            ['2','5',0.5],
            ['2','6',0.5],
            ['3','4',0.5],
            ['3','5',0],
            ['3','6',0.5],
            ],[A])

        s1 = Node(A, name="A")
        s2 = Node(B, name="B")
        s3 = Node(C, name="C")

        model = BayesianNetwork("tree")
        model.add_states(s1, s2, s3)
        model.add_edge(s1, s2)
        model.add_edge(s1, s3)
        model.bake()
        self.model = model

        meta = []
        for i in range(self.model.node_count()-1):
            meta.append({
                "name": chr(ord('A') + i),
                "type": "categorical",
                "size": 3,
                "i2s": ['1', '2', '3']
        })
        meta.append({
                "name": "C",
                "type": "categorical",
                "size": 3,
                "i2s": ['4', '5', '6']
        })
        self.meta = meta

class FCMaker(MultivariateMaker):
    def __init__(self):
        Rain = DiscreteDistribution({'T': 0.2, 'F': 0.8})
        Sprinkler = ConditionalProbabilityTable(
            [['F','T',0.4],
            ['F','F',0],
            ['T','T',0.1],
            ['T','F',0.9],
            ],[Rain])
        Wet = ConditionalProbabilityTable(
            [['F','F','T',0.01],
            ['F','F','F',0.99],
            ['F','T','T',0.8],
            ['F','T','F',0.2],
            ['T','F','T',0.9],
            ['T','F','F',0.1],
            ['T','T','T',0.99],
            ['T','T','F',0.01],
            ],[Sprinkler,Rain])

        s1 = Node(Rain, name="Rain")
        s2 = Node(Sprinkler, name="Sprinkler")
        s3 = Node(Wet, name="Wet")

        model = BayesianNetwork("Simple fully connected")
        model.add_states(s1, s2, s3)
        model.add_edge(s1, s2)
        model.add_edge(s1, s3)
        model.add_edge(s2, s3)
        model.bake()
        self.model = model

        meta = []
        for i in range(self.model.node_count()):
            meta.append({
                "name": None,
                "type": "categorical",
                "size": 2,
                "i2s": ['T', 'F']
        })
        meta[0]['name'] = 'Rain'
        meta[1]['name'] = 'Sprinkler'
        meta[2]['name'] = 'Wet'
        self.meta = meta


class GeneralMaker(MultivariateMaker):
    def __init__(self):
        L1 = DiscreteDistribution({'1': 0.4, '2': 0.1, '3':0.2, '4':0.3})
        L2 = ConditionalProbabilityTable(
            [['1','A',0.7],
            ['2','A',0.1],
            ['3','A',0.1],
            ['4','A',0],
            ['1','B',0],
            ['2','B',0.1],
            ['3','B',0.2],
            ['4','B',0.6],
            ],[L1]) 
        L3 = ConditionalProbabilityTable(
            [['A','1',0.1],
            ['B','1',0.9],
            
            ['A','2',0.5],
            ['B','2',0.5],
            
            ['A','3',0.2],
            ['B','3',0.8],
            
            ['A','4',0.3],
            ['B','4',0.7],
            
            ['A','5',0.5],
            ['B','5',0.5],
            
            ['A','6',0.15],
            ['B','6',0.85],
            
            ['A','7',0.7],
            ['B','7',0.3],

            ['A','8',0.5],
            ['B','8',0.5],

            ['A','9',0.6],
            ['B','9',0.4],

            ['A','10',0.3],
            ['B','10',0.7],
            
            ['A','11',0.8],
            ['B','11',0.2]
            ],[L2])
        L4 = ConditionalProbabilityTable(
            [['1','1',0.9],
            ['2','1',0.05],
            ['3','1',0.05],
            ['4','1',0],
            ['5','1',0],
            ['6','1',0],
            ['7','1',0],
            ['8','1',0],
            ['9','1',0],
            ['10','1',0],
            ['11','1',0],
            ['1','2',0.01],
            ['2','2',0.02],
            ['3','2',0.07],
            ['4','2',0.9],
            ['5','2',0],
            ['6','2',0],
            ['7','2',0],
            ['8','2',0],
            ['9','2',0],
            ['10','2',0],
            ['11','2',0],
            ['1','3',0],
            ['2','3',0],
            ['3','3',0],
            ['4','3',0],
            ['5','3',0.1],
            ['6','3',0.2],
            ['7','3',0.1],
            ['8','3',0.6],
            ['9','3',0],
            ['10','3',0],
            ['11','3',0],
            ['1','4',0],
            ['2','4',0],
            ['3','4',0],
            ['4','4',0],
            ['5','4',0],
            ['6','4',0],
            ['7','4',0],
            ['8','4',0],
            ['9','4',0.4],
            ['10','4',0.3],
            ['11','4',0.3]
            ],[L3])
        L5 = ConditionalProbabilityTable(
            [['1','1',0.65],
            ['2','1',0.25],
            ['3','1',0.05],
            ['4','1',0.05],
            ['1','2',0.01],
            ['2','2',0.01],
            ['3','2',0.9],
            ['4','2',0.08],
            ['1','3',0.6],
            ['2','3',0.05],
            ['3','3',0.05],
            ['4','3',0.3]
            ],[L4])
        L6 = ConditionalProbabilityTable(
            [['1','1',0.7],
            ['2','1',0.2],
            ['3','1',0.1],
            ['1','2',0.1],
            ['2','2',0.9],
            ['3','2',0],
            ['1','3',0.1],
            ['2','3',0.05],
            ['3','3',0.85]
            ],[L5])  
        L7 = ConditionalProbabilityTable(
            [['1','1',0.9],
            ['2','1',0],
            ['3','1',0.1],
            ['1','2',0.4],
            ['2','2',0.6],
            ['3','2',0],
            ['1','3',0],
            ['2','3',0.05],
            ['3','3',0.95],
            ['1','4',0.3],
            ['2','4',0.1],
            ['3','4',0.6]
            ],[L6])
        L8 = ConditionalProbabilityTable(
            [['1','1',0.7],
            ['2','1',0.2],
            ['3','1',0.1],
            ['4','1',0],
            ['1','2',0.1],
            ['2','2',0.4],
            ['3','2',0],
            ['4','2',0.5],
            ['1','3',0],
            ['2','3',0.05],
            ['3','3',0.85],
            ['4','3',0.1]
            ],[L7]) 
        L9 = ConditionalProbabilityTable(
            [['1','1',0.1],
            ['2','1',0.1],
            ['3','1',0.1],
            ['1','2',0.1],
            ['2','2',0.3],
            ['3','2',0.6],
            ],[L8])     
        s1 = Node(L1, name="L1")
        s2 = Node(L2, name="L2")
        s3 = Node(L3, name="L3")
        s4 = Node(L4, name="L4")
        s5 = Node(L5, name="L5")
        s6 = Node(L6, name="L6")
        s7 = Node(L7, name="L7")
        s8 = Node(L8, name="L8")
        s9 = Node(L9, name="L9")

        model = BayesianNetwork("Lung Cancer")
        model.add_states(s1, s2, s3, s4, s5, s6, s7, s8, s9)
        model.add_edge(s1, s2)
        model.add_edge(s2, s3)
        model.add_edge(s3, s4)
        model.add_edge(s4, s5)
        model.add_edge(s5, s6)
        model.add_edge(s6, s7)
        model.add_edge(s7, s8)
        model.add_edge(s8, s9)
        model.bake()
        self.model = model

        meta = []
        name_mapper = ["", "Smoker", "Cancer", "XRay", "Dyspnoea"]
        meta.append({
                "name": 'L1',
                "type": "categorical",
                "size": 4,
                "i2s": ['1', '2', '3', '4']
        })
        meta.append({
                "name": 'L2',
                "type": "categorical",
                "size": 2,
                "i2s": ['A', 'B']
        })
        meta.append({
                "name": 'L3',
                "type": "categorical",
                "size": 11,
                "i2s": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        })
        meta.append({
                "name": 'L4',
                "type": "categorical",
                "size": 4,
                "i2s": ['1', '2', '3', '4']
        })
        meta.append({
                "name": 'L5',
                "type": "categorical",
                "size": 3,
                "i2s": ['1', '2', '3']
        })
        meta.append({
                "name": 'L6',
                "type": "categorical",
                "size": 3,
                "i2s": ['1', '2', '3']
        })
        meta.append({
                "name": 'L7',
                "type": "categorical",
                "size": 4,
                "i2s": ['1', '2', '3', '4']
        })
        meta.append({
                "name": 'L8',
                "type": "categorical",
                "size": 3,
                "i2s": ['1', '2', '3']
        })
        meta.append({
                "name": 'L9',
                "type": "categorical",
                "size": 2,
                "i2s": ['1', '2']
        })
        self.meta = meta
