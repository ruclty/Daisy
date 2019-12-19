# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:35:43 2019

@author: susie
"""
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import math
import field

class VAE(nn.Module):
    def __init__(self,input_dim,encoder_dim,encoder_out_dim,latent_dim,decoder_dim,sample_rows,device,fields):
        super(VAE, self).__init__()
        self.fields = fields
        self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoder_dim),
                nn.BatchNorm1d(encoder_dim),
                nn.ReLU(inplace=False), 
                
                nn.Linear(encoder_dim, encoder_out_dim),
                nn.BatchNorm1d(encoder_out_dim),
                nn.ReLU(inplace=False)
                )
        self.fcmu = nn.Linear(encoder_out_dim, latent_dim) #mean
        self.fcsigma = nn.Linear(encoder_out_dim, latent_dim) #logsigma
        self.decoder = nn.Sequential(              
                nn.Linear(latent_dim, encoder_out_dim),
                nn.BatchNorm1d(encoder_out_dim),
                nn.ReLU(inplace = False),
                
                nn.Linear(encoder_out_dim,decoder_dim),
                nn.BatchNorm1d(decoder_dim),
                nn.ReLU(inplace = False),
                nn.Linear(decoder_dim, input_dim),
                nn.Sigmoid()
                )
        self.device = device
        self.sample_rows = sample_rows
        

    def reparameterize(self,mu,logvar):
        if self.device.type == 'cpu':
            epsi = Variable(torch.randn(mu.size(0), mu.size(1)))
        else:
            epsi = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
        z = mu + epsi*torch.exp(logvar/2)
        return z
    
    def forward(self,x):
        out1 = self.encoder(x) 
        mu = self.fcmu(out1) 
        logvar = self.fcsigma(out1) 
        z = self.reparameterize(mu, logvar)
        decode1 = self.decoder(z)
        current_ind = 0
        decodes = []
        for i in range(len(self.fields)):
            if self.fields[i].dtype == "Categorical Data":
                dim = self.fields[i].dim()
                decodes.append(fun.softmax(decode1[:,current_ind:current_ind+dim],dim=1))
                current_ind = current_ind+dim    
            else: 
                decodes.append(decode1[:,current_ind:current_ind+1])
                current_ind = current_ind +1
        decodes = torch.cat(decodes,dim=1)
        return decodes, mu, logvar

def loss_func(fields,reconstruct, x, mu, logvar):
    batch_size = x.size(0)
    MSE = 0
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/ batch_size
    BCE = 0
    curr = 0
    for i in range(len(fields)):
        dim = fields[i].dim()
        if fields[i].dtype == 'Numerical Data':
            mse = fun.mse_loss(reconstruct[:,curr:curr+dim], x[:,curr:curr+dim],reduction='mean')
            MSE += mse
        else:
            bce = fun.binary_cross_entropy(reconstruct[:,curr:curr+dim], x[:,curr:curr+dim], reduction='mean') 
            BCE += bce
        curr += dim
    return MSE,KLD,BCE

def alpha_schedule(epoch, max_epoch, alpha_max, strategy="exp"):
    # strategy to adjust weight
    if strategy == "linear":
        alpha = alpha_max * min(1, epoch / max_epoch)
    elif strategy == "exp":
        alpha = alpha_max * math.exp(-5 * (1 - min(1, epoch / max_epoch)) ** 2)
    else:
        raise NotImplementedError("Strategy {} not implemented".format(strategy))
    return alpha

def vaetrain(path, vae, data,fields, epochs, lr, steps_per_epoch, device):
    if device == torch.device('cuda'):
        vae.cuda()
    optimizer = optim.Adam(vae.parameters(),lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    for epoch in range(int(epochs)):
        alpha = alpha_schedule(epoch,800,0.5)
        vae.train()
        print("----------pretrain Epoch %d:----------\n"%epoch)
        log = open(path+"train_log_vae.txt","a+")
        log.write("----------pretrain Epoch %d:----------\n"%epoch)
        log.close()
        it = 0
        col_num = data.shape[1]
        while it < steps_per_epoch:
            dtest = data.values
            dtest = torch.FloatTensor(dtest)
            x = dtest
            if device == torch.device('cuda'):
                x = Variable(x).cuda()
            else:
                x = Variable(x)
            optimizer.zero_grad()
            x_, mu, logvar = vae.forward(x)
            MSE_loss, kl_loss,recon_loss = loss_func(fields,x_, x, mu, logvar)              
            elbo_loss = MSE_loss + recon_loss + alpha*kl_loss
            elbo_loss.backward()
            optimizer.step()
                
            if it%200 == 0:
                if device == torch.device('cuda'):
                    sample = vae.forward(torch.FloatTensor(torch.randn(vae.sample_rows, col_num)).cuda())[0].cpu()
                else:
                    sample = vae.forward(torch.FloatTensor(torch.randn(vae.sample_rows, col_num)))[0]
                sample_data = []
                current_ind = 0
                for i in range(len(fields)): 
                    dim = fields[i].dim()
                    data_totranse = sample[:,current_ind:(current_ind+dim)].detach().numpy()
                    sample_data.append(pd.DataFrame(fields[i].reverse(data_totranse)))
                    current_ind = current_ind+ dim
                sample_data = pd.concat(sample_data,axis=1)
                sample_data.to_csv(path+'sample_data_vae_{}_{}.csv'.format(epoch,it), index = None)
                if type(MSE_loss) == int or type(recon_loss)==int:
                    train_text = "VAE iteration {} \t ELBO Loss {elbo_loss:.4f} \t KL Loss {kl_loss:.4f}".format(
                        it,
                        elbo_loss=elbo_loss.item(),
                        kl_loss=kl_loss.item())
                else:
                    train_text = "VAE iteration {} \t ELBO Loss {elbo_loss:.4f} \t MSE Reconstruct Loss {mse_loss:.4f}\t BCE Reconstruct Loss {reconstruct_loss:.4f}\t KL Loss {kl_loss:.4f}".format(
                        it,
                        elbo_loss=elbo_loss.item(),
                        mse_loss=MSE_loss.item(),#连续
                        reconstruct_loss=recon_loss.item(),#离散
                        kl_loss=kl_loss.item())
                print(train_text)
                log = open(path+"train_log_vae.txt","a+")
                log.write(train_text)
                log.close()  
            it =it+ 1
            if it >= steps_per_epoch:
                break
    
