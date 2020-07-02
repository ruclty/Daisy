import pandas as pd
import numpy as np
import random
from pandas.api.types import is_string_dtype


def label_unlabel_split(data, fraction):
    #  data是需要被切分的含label的全体数据，格式为DataFrame
    #  frac是有标注数据集所占比例，值域[0, 1]
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Please input the data in type of DataFrame.")
    if fraction < 0 or fraction > 1 or not isinstance(fraction, float):
        raise ValueError("Please input a fraction between 0 and 1 for fraction.")

    # 按照fraction的比例切分有标注数据集和无标注数据集
    label_data = data.sample(frac=fraction, axis=0)
    unlabel_data = data.loc[data.index.difference(label_data.index)]

    # 返回有标注数据集和无标注数据集，格式均为DataFrame
    return label_data, unlabel_data



def MCAR(data, preplace, pmissing, label_column):
    #  data是需要添加噪音的含label的数据，格式为DataFrame
    #  preplace是每个属性被替换的概率，值域[0, 1]
    #  pmissing是每个属性被清除的概率，值域[0, 1]
    #  label_column是数据中为label的列名，应当在已有列之中
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Please input the data in type of DataFrame.")
    if preplace < 0 or preplace > 1 or not isinstance(preplace, float):
        raise ValueError("Please input a possibility between 0 and 1 for preplace.")
    if pmissing < 0 or pmissing > 1 or not isinstance(pmissing, float):
        raise ValueError("Please input a possibility between 0 and 1 for pmissing.")
    if label_column not in data.columns:
        raise ValueError("Please input one column name of label correctly.")

    # 计算需要被清除或替换的样本数
    count_replace = int(preplace * len(data))
    count_missing = int(pmissing * len(data))
    count_all = int(count_missing + count_replace)

    noise_data = data.copy()
    for column in data.columns:
        # label_column不做处理
        if column == label_column:
            continue

        # 生成需要被清除或替换的样本索引号
        length = list(range(len(data)))
        index_all = random.sample(length, k=count_all)
        index_replace = index_all[:count_replace]
        index_missing = index_all[count_replace:]

        # 将对应样本的column属性清除
        print("column ", column, " is under the process of missing...")
        noise_data.loc[index_missing, column] = np.nan


        # 将对应样本的column属性替换
        print("column ", column, " is under the process of replacing...")
        noise_data.loc[index_replace, column] = np.random.choice(list(data[column]), count_replace)

    # 返回原数据和添加噪音后的数据，格式均为DataFrame
    return data, noise_data


def certain_na_drop(data, string):
    #  data是需要清洗的含label的数据，格式为DataFrame
    #  string是在该data中代表空值的符号的正则表达式
    #  如census数据集中含有“？”的为空值，则此处string为'\?'

    clean_data = data.copy()
    for column in data.columns:
        if is_string_dtype(data[column]):  # 如果不是string的列，str方法会报错
            clean_data = clean_data[~(clean_data[column].str.contains(string))]

    # 返回原数据和去除含空值样本后的数据，格式均为DataFrame
    return data, clean_data


