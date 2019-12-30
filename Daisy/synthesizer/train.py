import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
import math

# compute kl loss (not use now)    
def compute_kl(real, pred):
    return torch.sum((torch.log(pred + 1e-4) - torch.log(real + 1e-4)) * pred)

def KL_Loss(x_fake, x_real, col_type, col_dim):
    kl = 0.0
    sta = 0
    end = 0
    for i in range(len(col_type)):
        dim = col_dim[i]
        sta = end
        end = sta+dim
        fakex = x_fake[:,sta:end]
        realx = x_real[:,sta:end]
        if col_type[i] == "gmm":
            fake2 = fakex[:,1:]
            real2 = realx[:,1:]
            dist = torch.sum(fake2, dim=0)
            dist = dist / torch.sum(dist)
            real = torch.sum(real2, dim=0)
            real = real / torch.sum(real)
            kl += compute_kl(real, dist)
        else:
            dist = torch.sum(fakex, dim=0)
            dist = dist / torch.sum(dist)
            
            real = torch.sum(realx, dim=0)
            real = real / torch.sum(real)
            
            kl += compute_kl(real, dist)
    return kl
    
def V_Train(t, path, sampleloader, G, D, epochs, lr, dataloader, z_dim, dataset, col_type,itertimes = 100, steps_per_epoch = None, GPU=False, KL=True, ratio=1):
    """
    The vanilla (basic) training process for GAN
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sample_rows: # of synthesized rows
        * G: the generator
        * D: the discriminator
        * epochs: # of epochs
        * lr: learning rate
        * dataloader: the data loader
        * z_dim: dimension of noise
        * dataset: the dataset for reversible data tranformation
        * itertimes:
        * steps_per_epoch: # of steps per epoch
        * GPU: the GPU flag
    Return:
        * G: the generator
        * D: the descriminator
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True
    
    D_optim = optim.Adam(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.Adam(G.parameters(), lr=lr, weight_decay=0.00001)

    # the default # of steps is the # of batches.
    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)

    for epoch in range(epochs):
        it = 0
        log = open(path+"train_log_"+str(t)+".txt","a+")
        log.write("-----------Epoch {}-----------\n".format(epoch))
        log.close()
        print("-----------Epoch {}-----------".format(epoch))
        while it < steps_per_epoch:
            for x_real in dataloader:
                if GPU:
                    x_real = x_real.cuda()

                ''' train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                
                x_fake = G(z)

                y_real = D(x_real)
                y_fake = D(x_fake)
                
                # D_Loss = -(torch.mean(y_real) - torch.mean(y_fake)) # wgan loss
                fake_label = torch.zeros(y_fake.shape[0], 1)
                real_label = np.ones([y_real.shape[0], 1])
                # Avoid the suppress of Discriminator over Generator
                real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
                real_label = torch.from_numpy(real_label).float()
                if GPU:
                    fake_label = fake_label.cuda()
                    real_label = real_label.cuda()
                    
                D_Loss1 = F.binary_cross_entropy(y_real, real_label)
                D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
                D_Loss = D_Loss1 + D_Loss2
                
                G_optim.zero_grad()
                D_optim.zero_grad()
                D_Loss.backward()
                D_optim.step()

                ''' train Generator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                
                x_fake = G(z)
                y_fake = D(x_fake)
                
                real_label = torch.ones(y_fake.shape[0], 1)
                if GPU:
                    real_label = real_label.cuda()
                G_Loss1 = F.binary_cross_entropy(y_fake, real_label)
                if KL:
                    KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
                    G_Loss = G_Loss1 + KL_loss
                else:
                    G_Loss = G_Loss1

                G_optim.zero_grad()
                D_optim.zero_grad()
                G_Loss.backward()
                G_optim.step()

                it += 1

                if it%itertimes == 0:
                    log = open(path+"train_log_"+str(t)+".txt","a+")
                    log.write("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))
                    log.close()
                    print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))
                if it >= steps_per_epoch:
                    G.eval()
                    if GPU:
                        G.cpu()
                        G.GPU = False
                    z = torch.randn(int(len(sampleloader.data)*ratio), z_dim)
                    x_fake = G(z)
                    samples = x_fake.cpu()
                    samples = samples.reshape(samples.shape[0], -1)
                    samples = samples[:,:dataset.dim]
                    sample_table = dataset.reverse(samples.detach().numpy())
                    sample_data = pd.DataFrame(sample_table,columns=dataset.columns)
                    sample_data.to_csv(path+'sample_data_{}_{}.csv'.format(t,epoch), index = None)
                    if GPU:
                        G.cuda()
                        G.GPU = True
                    G.train()
                    break
    return G,D


def W_Train(t, path, sampleloader, G, D, ng, nd, cp, lr, dataloader, z_dim, dataset, col_type,itertimes = 100, GPU=False, KL=True):
    """
    The WGAN training process for GAN
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sample_rows: # of synthesized rows
        * G: the generator
        * D: the discriminator
        * ng: 
        * nd:
        * cp:
        * lr: learning rate
        * dataloader: the data loader
        * z_dim: dimension of noise
        * dataset: the dataset for reversible data tranformation
        * itertimes:
        * steps_per_epoch: # of steps per epoch
        * GPU: the GPU flag
    Return:
        * G: the generator
        * D: the descriminator
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True
    
    D_optim = optim.RMSprop(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.RMSprop(G.parameters(), lr=lr, weight_decay=0.00001)
        
    epoch_time = int(ng/10)
    # the default # of steps is the # of batches.

    for t1 in range(ng):
        for t2 in range(nd):
            x_real = dataloader.sample(dataloader.batch_size)
            if GPU:
                x_real = x_real.cuda()

            ''' train Discriminator '''
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                z = z.cuda()
            
            x_fake = G(z)

            y_real = D(x_real)
            y_fake = D(x_fake)
                
            D_Loss = -(torch.mean(y_real) - torch.mean(y_fake))
            
            G_optim.zero_grad()
            D_optim.zero_grad()
            D_Loss.backward()
            D_optim.step()
            for p in D.parameters():
                p.data.clamp_(-cp, cp)  # clip the discriminator parameters (wgan)

        ''' train Generator '''
        z = torch.randn(dataloader.batch_size, z_dim)
        if GPU:
            z = z.cuda()
        x_fake = G(z)
        y_fake = D(x_fake)
        G_Loss1 = -torch.mean(y_fake)
        if KL:
            KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
            G_Loss = G_Loss1 + KL_loss
        else:
            G_Loss = G_Loss1
        G_optim.zero_grad()
        D_optim.zero_grad()
        G_Loss.backward()
        G_optim.step()

        if t1 % itertimes == 0:    
            print("------ng : {}-------".format(t1))
            print("generator loss: {}".format(G_Loss.data))
            print("discriminator loss: {}".format(D_Loss.data))
            log = open(path+"train_log_"+str(t)+".txt","a+")
            log.write("----------ng: {}---------\n".format(t1))
            log.write("generator loss: {}\n".format(G_Loss.data))
            log.write("discriminator loss: {}\n".format(D_Loss.data))
            log.close()  
        if t1 % epoch_time == 0 and t1 > 0:
            G.eval()
            if GPU:
                G.cpu()
                G.GPU = False
            z = torch.randn(len(sampleloader.data), z_dim)
            x_fake = G(z)          
            samples = x_fake.cpu()
            samples = samples.reshape(samples.shape[0], -1)
            samples = samples[:,:dataset.dim]
            sample_table = dataset.reverse(samples.detach().numpy())
            sample_data = pd.DataFrame(sample_table,columns=dataset.columns)
            sample_data.to_csv(path+'sample_data_{}_{}.csv'.format(t,int(t1/epoch_time)), index = None)
            if GPU:
                G.cuda()
                G.GPU = True
            G.train()
    return G,D


def C_Train(t, path, sampleloader, G, D, epochs, lr, dataloader, z_dim, dataset, col_type,itertimes = 100, steps_per_epoch = None, GPU=False):
    """
    The
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sampleloader:
    :param G:
    :param D:
    :param epochs:
    :param lr:
    :param dataloader:
    :param z_dim:
    :param dataset:
    :param itertimes:
    :param steps_per_epoch:
    :param GPU:
    :return:
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True
    all_labels = dataloader.label
    conditions = np.unique(all_labels.view(all_labels.dtype.descr * all_labels.shape[1]))
    D_optim = optim.Adam(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.Adam(G.parameters(), lr=lr, weight_decay=0.00001)    
    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)
    for epoch in range(epochs):
        log = open(path+"train_log_"+str(t)+".txt","a+")
        log.write("-----------Epoch {}-----------\n".format(epoch))
        log.close()
        print("-----------Epoch {}-----------".format(epoch))
        for it in range(steps_per_epoch):
            for c in conditions:
                x_real, c_real = dataloader.sample(label=list(c))
                if GPU:
                    x_real = x_real.cuda()
                    c_real = c_real.cuda()
                ''' train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                    
                x_fake = G(z, c_real)
                y_real = D(x_real, c_real)
                y_fake = D(x_fake, c_real)
                
                #D_Loss = -(torch.mean(y_real) - torch.mean(y_fake)) # wgan loss
                fake_label = torch.zeros(y_fake.shape[0], 1)
                real_label = np.ones([y_real.shape[0], 1])
                real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
                real_label = torch.from_numpy(real_label).float()
                if GPU:
                    fake_label = fake_label.cuda()
                    real_label = real_label.cuda()
                
                D_Loss1 = F.binary_cross_entropy(y_real, real_label)
                D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
                D_Loss = D_Loss1 + D_Loss2
                
                G_optim.zero_grad()
                D_optim.zero_grad()
                D_Loss.backward()
                D_optim.step()
                ''' train Generator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                    
                x_fake = G(z, c_real)
                y_fake = D(x_fake, c_real)
                
                real_label = torch.ones(y_fake.shape[0], 1)
                if GPU:
                    real_label = real_label.cuda()
                    
                G_Loss1 = F.binary_cross_entropy(y_fake, real_label)
                KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
                G_Loss = G_Loss1 + KL_loss

                G_optim.zero_grad()
                D_optim.zero_grad()
                G_Loss.backward()
                G_optim.step()

            if it%itertimes == 0:
                log = open(path+"train_log_"+str(t)+".txt","a+")
                log.write("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))
                log.close()
                print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))

        G.eval()
        if GPU:
            G.cpu()
            G.GPU = False
        y = torch.from_numpy(sampleloader.label).float()
        z = torch.randn(len(sampleloader.label), z_dim)
        x_fake = G(z, y)
        x_fake = torch.cat((x_fake, y), dim = 1)
        samples = x_fake.cpu()
        samples = samples.reshape(samples.shape[0], -1)
        samples = samples[:,:dataset.dim]
        sample_table = dataset.reverse(samples.detach().numpy())
        sample_data = pd.DataFrame(sample_table,columns=dataset.columns)
        sample_data.to_csv(path+'sample_data_{}_{}.csv'.format(t,epoch), index = None)
        if GPU:
            G.cuda()
            G.GPU = True
        G.train()
    return G,D

def C_Train_nofair(t, path, sampleloader, G, D, epochs, lr, dataloader, z_dim, dataset, col_type,itertimes = 100, steps_per_epoch = None, GPU=False):
    """
    The
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sampleloader:
    :param G:
    :param D:
    :param epochs:
    :param lr:
    :param dataloader:
    :param z_dim:
    :param dataset:
    :param itertimes:
    :param steps_per_epoch:
    :param GPU:
    :return:
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True
    all_labels = dataloader.label
    conditions = np.unique(all_labels.view(all_labels.dtype.descr * all_labels.shape[1]))
    D_optim = optim.Adam(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.Adam(G.parameters(), lr=lr, weight_decay=0.00001)
        
    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)
    for epoch in range(epochs):
        log = open(path+"train_log_"+str(t)+".txt","a+")
        log.write("-----------Epoch {}-----------\n".format(epoch))
        log.close()
        print("-----------Epoch {}-----------".format(epoch))
        it = 0
        while it < steps_per_epoch:
            for x_real, c_real in dataloader:
                if GPU:
                    x_real = x_real.cuda()
                    c_real = c_real.cuda()
                ''' train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()           
                x_fake = G(z, c_real)
                y_real = D(x_real, c_real)
                y_fake = D(x_fake, c_real)
                fake_label = torch.zeros(y_fake.shape[0], 1)
                real_label = np.ones([y_real.shape[0], 1])
                real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
                real_label = torch.from_numpy(real_label).float()
                if GPU:
                    fake_label = fake_label.cuda()
                    real_label = real_label.cuda()
                
                D_Loss1 = F.binary_cross_entropy(y_real, real_label)
                D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
                D_Loss = D_Loss1 + D_Loss2
                
                G_optim.zero_grad()
                D_optim.zero_grad()
                D_Loss.backward()
                D_optim.step()
                ''' train Generator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                    
                x_fake = G(z, c_real)
                y_fake = D(x_fake, c_real)
                
                real_label = torch.ones(y_fake.shape[0], 1)
                if GPU:
                    real_label = real_label.cuda()
                    
                G_Loss1 = F.binary_cross_entropy(y_fake, real_label)
                KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
                G_Loss = G_Loss1 + KL_loss
                G_optim.zero_grad()
                D_optim.zero_grad()
                G_Loss.backward()
                G_optim.step()
                it += 1

                if it%itertimes == 0:
                    log = open(path+"train_log_"+str(t)+".txt","a+")
                    log.write("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))
                    log.close()
                    print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))
                    
                if it >= steps_per_epoch:
                    G.eval()
                    if GPU:
                        G.cpu()
                        G.GPU = False
                    y = torch.from_numpy(sampleloader.label).float()
                    z = torch.randn(len(sampleloader.label), z_dim)
                    x_fake = G(z, y)
                    x_fake = torch.cat((x_fake, y), dim = 1)
                    samples = x_fake.cpu()
                    samples = samples.reshape(samples.shape[0], -1)
                    samples = samples[:,:dataset.dim]
                    sample_table = dataset.reverse(samples.detach().numpy())
                    sample_data = pd.DataFrame(sample_table,columns=dataset.columns)
                    sample_data.to_csv(path+'sample_data_{}_{}.csv'.format(t,epoch), index = None)
                    if GPU:
                        G.cuda()
                        G.GPU = True
                    G.train()
                    break
    return G,D


def C_Train_dp(t, path, sampleloader,G, D, ng, nd, cp, lr, dataloader, z_dim, dataset, col_type, eps, itertimes = 100, GPU=False,delta=0.00001):
    """
    The Conditional Training with Differential Privacy
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sampleloader: 
        * G: the generator
        * D: the discriminator
        * ng: 
        * nd:
        * cp:
        * lr: learning rate
        * dataloader: the data loader
        * z_dim: dimension of noise
        * dataset: the dataset for reversible data tranformation
        * itertimes:
        * steps_per_epoch: # of steps per epoch
        * GPU: the GPU flag
    Return:
        * G: the generator
        * D: the descriminator
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True
    all_labels = dataloader.label
    conditions = np.unique(all_labels.view(all_labels.dtype.descr * all_labels.shape[1]))
    D_optim = optim.RMSprop(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.RMSprop(G.parameters(), lr=lr, weight_decay=0.00001)
    
    q = dataloader.batch_size / len(dataloader.data)
    theta_n = 2*q*math.sqrt(nd*math.log(1/delta)) / eps     
    epoch_time = int(ng/5)
    print("theta_n: {}".format(theta_n))
    # the default # of steps is the # of batches.

    for t1 in range(ng):
        for c in conditions:
            for t2 in range(nd):
                x_real, c_real = dataloader.sample(label=list(c))
                if GPU:
                    x_real = x_real.cuda()
                    c_real = c_real.cuda()
                ''' train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()

                x_fake = G(z, c_real)
                      
                y_real = D(x_real, c_real)
                y_fake = D(x_fake, c_real)
                    
                D_Loss = -(torch.mean(y_real) - torch.mean(y_fake))
                
                D_optim.zero_grad()
                G_optim.zero_grad()
                D_Loss.backward()
                
                for p in D.parameters():
                    sigma = theta_n * 1
                    noise = np.random.normal(0, sigma, p.grad.shape) / dataloader.batch_size
                    noise = torch.from_numpy(noise).float()
                    if GPU:
                        noise = noise.cuda()
                    p.grad += noise
                
                D_optim.step()
                for p in D.parameters():
                    p.data.clamp_(-cp, cp)  # clip the discriminator parameters (wgan)

            ''' train Generator '''
            x_real, c_real = dataloader.sample(label=list(c))
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                z = z.cuda()
                c_real = c_real.cuda()
            x_fake = G(z, c_real)
            y_fake = D(x_fake, c_real)
            G_Loss = -torch.mean(y_fake)
           # KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
            #G_Loss = G_Loss1 + KL_loss
            G_optim.zero_grad()
            D_optim.zero_grad()
            G_Loss.backward()
            G_optim.step()

        if t1 % itertimes == 0:    
            print("------ng : {}-------".format(t1))
            print("generator loss: {}".format(G_Loss.data))
            print("discriminator loss: {}".format(torch.mean(D_Loss).data))
            log = open(path+"train_log_"+str(t)+".txt","a+")
            log.write("----------ng: {}---------\n".format(t1))
            log.write("generator loss: {}\n".format(G_Loss.data))
            log.write("discriminator loss: {}\n".format(torch.mean(D_Loss).data))
            log.close()  
        
        if (t1+1) % epoch_time == 0 and t1 > 0:
            G.eval()
            if GPU:
                G.cpu()
                G.GPU = False
            y = torch.from_numpy(sampleloader.label).float()
            z = torch.randn(len(sampleloader.label), z_dim)
            x_fake = G(z, y)
            x_fake = torch.cat((x_fake, y), dim = 1)
            samples = x_fake.cpu()
            samples = samples.reshape(samples.shape[0], -1)
            samples = samples[:,:dataset.dim]
            sample_table = dataset.reverse(samples.detach().numpy())
            sample_data = pd.DataFrame(sample_table,columns=dataset.columns)
            sample_data.to_csv(path+'sample_data_{}_{}_{}.csv'.format(eps, t,int(t1/epoch_time)), index = None)
            if GPU:
                G.cuda()
                G.GPU = True
            G.train()
    return G,D
        
        
        
        
