# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:13:29 2019

@author: susie
"""

import pandas as pd
import numpy as np
import torch
import argparse
import json
import os
from model_new import vaetrain
from model_new import VAE
from field import CategoricalField, NumericalField


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='a json config file',default ='C:/Users/58454/Desktop/GAN/param-sdata2.json')
    args = parser.parse_args()
    with open(args.config) as f:
        param = json.load(f)
    try:
        os.mkdir("expdir")
    except:
        pass
    path = "expdir/"+param["name"]+"/"
    try:
        os.mkdir("expdir/"+param["name"])
    except:
        pass
    encoder_dim = param["encoder_dim"]
    encoder_out_dim = param["encoder_out_dim"]
    decoder_dim = param["decoder_dim"]
    latent_dim = param["latent_dim"]
    sample_rows = param["sample_rows"]
    lr = param["lr"]
    header = param["header"]
    epochs = param["epochs"]
    steps_per_epoch = param["steps_per_epoch"]
    if header == 1:
        data = pd.read_csv(param["train"])
    else:
        data = pd.read_csv(param["train"], header=None)
    data.columns = [x for x in range(data.shape[1])]
    cuda = not param["no_cuda"] and torch.cuda.is_available()
    model_name = param['model_name']
    fields = []
    feed_data = []
    for i, col in enumerate(list(data)):
        if i in param["continuous_cols"]:
            if model_name == "minmax":
                col2 = NumericalField(model='minmax')
            elif model_name == "meanstd":
                col2 = NumericalField(model='meanstd')
            col2.get_data(data[i])
            fields.append(col2)
            col2.learn()
            feed_data.append(col2.convert(np.asarray(data[i])))
        else:
            col1 = CategoricalField("one-hot", noise=None)
            fields.append(col1)
            col1.get_data(data[i])
            col1.learn()
            features = col1.convert(np.asarray(data[i]))
            cols = features.shape[1]
            rows = features.shape[0]
            for j in range(cols):
                feed_data.append(features.T[j])
    feed_data = pd.DataFrame(feed_data).T
    input_dim = feed_data.shape[1]
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    vae = VAE(input_dim,encoder_dim,encoder_out_dim,latent_dim,decoder_dim,sample_rows,device,fields).to(device)
    vaetrain(path,  vae, feed_data, fields, epochs, lr, steps_per_epoch = steps_per_epoch,device = device)
