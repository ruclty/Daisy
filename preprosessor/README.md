# Preprocessor: data split and add noise
## version 1.0 2020/05/19


## label_unlabel_split(data, fraction)
### description
该函数按照给定的比例随机切分有标注数据集和无标注数据集
### input
* data是需要被切分的（含label）的全体数据，格式为DataFrame
* frac是有标注数据集所占比例，值域[0, 1]
### output
```
return label_data, unlabel_data
```
* 返回有标注数据集和无标注数据集，格式均为DataFrame

## MCAR(data, preplace, pmissing, label_column)
### description
对数据每一个属性，该函数按照给定的缺失概率进行随机替换或清除，给出增加噪音后的数据
### input
* data是需要添加噪音的含label的数据，格式为DataFrame
* pnoise是每个属性被替换的概率，值域[0, 1]
* pmissing是每个属性被清除的概率，值域[0, 1]
* label_column是数据中为label的列名，应当在已有列之中
### output
```
return data, noise_data
```
* 返回原数据和添加噪音后的数据，格式均为DataFrame
    
    


