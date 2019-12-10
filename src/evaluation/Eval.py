import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network import MLPClassifier
from PMF import PMF

class Eval_table(object):
    
    def __init__(self, csv_path, continuous_list=None, sample_num=None):
        self.csv_path = csv_path
        self.continuous_list = continuous_list
        self.csv_data = pd.read_csv(self.csv_path)
        self.csv_title = self.csv_data.columns.values.tolist()
        if sample_num is None:
            self.csv_sample_data = self.csv_data
        else:
            self.csv_sample_data = self.csv_data.sample(n=sample_num, axis=0)
    

    def resample(self, sample_num):
        self.csv_sample_data = self.csv_data.sample(n=sample_num, axis=0)


    def get_column_data(self, column, condition=None, value=None):
        if condition is not None:
            assert value is not None
            col = self.csv_sample_data.loc[(self.csv_sample_data[self.csv_title[condition]] == value), [self.csv_title[column]]].values.reshape((1,-1))[0]
        else:
            col = self.csv_sample_data.take([column], axis=1).values.reshape((1,-1))[0]
        return col


    def get_column_PMF(self, column, interval=1, condition=None, value=None):
        col = self.get_column_data(column, condition, value)
        if column in self.continuous_list:
            continuous = True
        else:
            continuous =  False
        pmf = PMF(col, continuous=continuous, interval=interval)
        return pmf

    
    def process_data(self, feature_names=None):
        self.labels = LabelBinarizer().fit_transform(self.csv_sample_data.take([-1], axis=1).values.reshape((1,-1))[0].tolist())
        length = len(self.csv_title)
        if feature_names is None:
            feature_list = []
            self.feature_names = []
            for i in range(length-1):
                col = self.csv_sample_data.take([i], axis=1).values.reshape((1,-1))[0]
                if i not in self.continuous_list:
                    df = pd.get_dummies(col, prefix=self.csv_title[i])
                    col = df.values
                    self.feature_names += df.columns.values.tolist()
                else:
                    self.feature_names.append(self.csv_title[i])
                feature_list.append(col)
            self.features = np.column_stack(feature_list)
        else:
            feature_list = []
            self.feature_names = feature_names
            for i in range(0, len(self.csv_sample_data)):
                row_dict = dict(zip(feature_names, np.zeros(len(feature_names), dtype=np.int)))
                for j in range(0, length-1):
                    if j in self.continuous_list:
                        row_dict[self.csv_title[j]] = self.csv_sample_data.iloc[i][self.csv_title[j]]
                    else:
                        row_dict[self.csv_title[j] + '_' + self.csv_sample_data.iloc[i][self.csv_title[j]]] = 1
                feature_list.append(list(row_dict.values()))
            self.features = np.array(feature_list)

                
    def model_evaluation(self, model=DecisionTreeClassifier(max_depth=20), test_data=None, test_label=None, ratio=0.7):
        if test_data is not None and test_label is not None:
            train_data, train_label = self.features, self.labels
        else:
            sum = len(self.csv_sample_data)
            train_num = int(sum * ratio)
            train_data, train_label = self.features[:train_num], self.labels[:train_num]
            test_data, test_label = self.features[train_num:], self.labels[train_num:]
        
        model.fit(train_data, train_label)
        predict_result = model.predict(test_data)
        
        return accuracy_score(test_label, predict_result), f1_score(test_label, predict_result), confusion_matrix(test_label, predict_result), roc_auc_score(test_label, predict_result)
        