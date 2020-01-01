# Daisy: Relational Data Synthesis using Generative Adversarial Networks

## Technical Report: Relational Data Synthesis using Generative Adversarial Networks: A Design Space Exploration

<https://github.com/ruclty/Daisy/blob/master/daisy.pdf>


## Requirements
Before running the code, please make sure your Python version is above **3.6**.
Then install necessary packages by :
```sh
pip3 install -r requirements.txt
```
## Datasets
We provide four datasets, which can be found in the `dataset` folder.

## Parameters
 You need write a .json file as the input parameters of code. The format of parameter file should be :
```json
[{
    "name": " required, name of the output file " ,
    "train": " required, path of the training file " ,
    "sample": " required, path of the sampling file " ,
    "normalize_cols": " required, a list contains index of the normalize columns " ,
    "gmm_cols": " required, a list contains index of the gmm columns " ,
    "one-hot_cols":" required, a list contains index of the one-hot columns " ,
    "ordinal_cols":" required, a list contains index of the ordinal columns " ,
    "model": " required, model of the generator, LGAN(LSTM) or VGAN(MLP) " ,
    "dis_model": " optional, model of the discriminator, lstm or mlp, default mlp ",
    "n_epochs": " required, num of training epochs " ,
    "steps_per_epoch": " required, steps per epoch " ,
    "n_search": " required, training times " ,
    "rand_search": " required, whether to search hyper-parameters randomly " , 
    "param": " required if rand_search is 'no', hyper-parameter of the NN " , 
    "train_method": " required, training method " ,  
    "label": " required if train_method is a conditional training, name of the label column " , 
    "KL": " optional, whether using KL loss in training, default 'yes' " , 
    "ratio" : " optional, the ratio of the number of sample records to the real data, default 1"
},
{
    "another parameters group ..."
},
"..."]
```
Folder "params" contains some examples, you can run the code using those parameter files directly, or write a self-defined parameter file to train a new dataset.

## Run
Run the code with the following command :
```sh
python Daisy/run.py [parameter file]
```
You can run the following command for quickly start :
```sh
./run.sh
```
