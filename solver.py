# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:26:29 2020

@author: XIANG
"""
import torch.optim as optim
import torch
import torch.nn as nn
from os.path import dirname, join as pjoin
from collections import OrderedDict
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from skimage.transform import radon, iradon

def l1_loss(pred, target, l1_weight):
    """
    Compute L1 loss;
    l1_weigh default: 0.1
    """
    err = torch.mean(torch.abs(pred - target))
    err = l1_weight * err
    return err

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss. 0.01 default.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


class Solver(object):
    def __init__(self, model, data_loader, args, test_data, test_images):
        assert args.model_name in ['full_net','denoise_net', 'MoDL', 'fista_net']

        self.model_name = args.model_name
        self.model = model
        self.data_loader = data_loader
        self.data_dir = args.data_dir
        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.lr = args.lr

        if self.model_name == 'MoDL':
            # set different learning rate for cnn and numerical block
            self.optimizer = optim.Adam([
                {'params': self.model.dw.parameters()}, 
                {'params': self.model.dc.parameters(), 'lr': 0.0001}
                ], lr=self.lr, weight_decay=0.0001)
        elif self.model_name == 'fista_net':
            # set different lr for regularization weights and network weights
            self.optimizer = optim.Adam([
            {'params': self.model.fcs.parameters()}, 
            {'params': self.model.w_theta, 'lr': 0.0001},
            {'params': self.model.b_theta, 'lr': 0.0001},
            {'params': self.model.w_mu, 'lr': 0.0001},
            {'params': self.model.b_mu, 'lr': 0.0001},
            {'params': self.model.w_rho, 'lr': 0.0001},
            {'params': self.model.b_rho, 'lr': 0.0001}], 
            lr=self.lr, weight_decay=0.0001)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), self.lr, weight_decay=0.0001)
        
        self.theta = args.theta
        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu
        self.use_cuda = args.use_cuda
        self.log_interval = args.log_interval
        self.test_epoch = args.test_epoch
        self.test_data = test_data
        self.test_images = test_images
        self.train_loss = nn.MSELoss()

    def save_model(self, iter_):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        f = pjoin(self.save_path, 'epoch_{}.ckpt'.format(iter_))
        torch.save(self.model.state_dict(), f)
    
    def load_model(self, iter_):
        f = pjoin(self.save_path, 'epoch_{}.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.model.load_state_dict(state_d)
        else:
            self.model.load_state_dict(torch.load(f))
            
    
    def train(self):
        train_losses = []
        start_time = time.time()
        # set up Tensorboard
        writer = SummaryWriter('runs/'+self.model_name)

        for epoch in range(1+self.start_epoch, self.num_epochs+1+self.start_epoch):
            self.model.train(True)
                
            for batch_idx, (x_in, y_target) in enumerate(self.data_loader):
                
                # add one channel
                b_in = torch.unsqueeze(x_in, 1) 
                y_target = torch.unsqueeze(y_target, 1)

                # do some conversions for different networks
                X_fbp = torch.zeros_like(y_target)
                batch_size = y_target.shape[0]
                for i in range(batch_size):
                    sino = b_in[i].squeeze()
                    X0 = iradon(sino, theta=self.theta)
                    X_fbp[i] = torch.from_numpy(X0)

                # move to gpu
                if self.use_cuda:
                    X_fbp = X_fbp.cuda().float()         # initial guess x0
                    #b_in = b_in.cuda().float()           # measured sinogram
                    y_target = y_target.cuda().float()   # ground truth image
                
                # predict and compute losses
                if self.model_name == 'MoDL':
                    pred = self.model(X_fbp, b_in)
                    loss = self.train_loss(pred, y_target) #+ l1_loss(pred, y_target, 0.3)
                if self.model_name == 'fista_net':
                    [pred, loss_layers_sym, encoder_st] = self.model(X_fbp, b_in)

                    # Compute loss, data consistency and regularizer constraints
                    loss_discrepancy = self.train_loss(pred, y_target) #+ l1_loss(pred, y_target, 0.3)
                    loss_constraint = 0
                    for k, _ in enumerate(loss_layers_sym, 0):
                        loss_constraint += torch.mean(torch.pow(loss_layers_sym[k], 2))

                    encoder_constraint = 0
                    for k, _ in enumerate(encoder_st, 0):
                        encoder_constraint += torch.mean(torch.abs(encoder_st[k]))

                    # loss = loss_discrepancy + gamma * loss_constraint
                    loss = loss_discrepancy +  0.01 * loss_constraint + 0.001 * encoder_constraint
                if self.model_name == 'denoise_net':
                    pred = self.model(X_fbp)
                    loss = self.train_loss(pred, y_target) #+ l1_loss(pred, y_target, 0.3)
                
                self.model.zero_grad()
                self.optimizer.zero_grad()
                
                # backpropagate the gradients
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                

                # print processes 
                if batch_idx % self.log_interval == 0:
                    writer.add_scalar('training loss', loss.data, epoch * len(self.data_loader) + batch_idx)

                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}\t TIME:{:.1f}s'
                          ''.format(epoch, batch_idx,
                                    len(self.data_loader),
                                    100. * batch_idx / len(self.data_loader),
                                    loss.data,
                                    time.time() - start_time))  

                    # print weight values of model
                    if self.model_name == 'fista_net':

                        print("Threshold value w: {}".format(self.model.w_theta))
                        print("Threshold value b: {}".format(self.model.b_theta))
                        print("Gradient step w: {}".format(self.model.w_mu))
                        print("Gradient step b: {}".format(self.model.b_mu))
                        print("Two step update w: {}".format(self.model.w_rho))
                        print("Two step update b: {}".format(self.model.b_rho))

                    if self.model_name == 'MoDL':
                        print("Regularization parameter: {}".format(self.model.dc.lam))
                      
  
            # save model
            if epoch % 1 == 0:

                self.save_model(epoch)
                np.save(pjoin(self.save_path, 'loss_{}_epoch.npy'.format(epoch)), np.array(train_losses))
        
    def test(self):
        self.load_model(self.test_epoch)
        self.model.eval()
        
        with torch.no_grad():
            # Must use the sample test dataset!!!
            X_fbp = torch.zeros_like(self.test_images)
            batch_size = self.test_images.shape[0]
            for i in range(batch_size):
                sino = self.test_data[i].squeeze()
                X0 = iradon(sino, theta=self.theta)
                X_fbp[i] = torch.from_numpy(X0)

            if self.model_name == "denoise_net":
                test_res = self.model(X_fbp.cuda().float())
            if self.model_name == "fista_net":
                [test_res, _] = self.model(X_fbp.cuda().float(), self.test_data.cuda().float())
                # torch.save(test_res, 'iteration_result.pt') # iteration result
                # test_res = test_res[5] # iteration result

            if self.model_name == "MoDL":
                test_res = self.model(X_fbp.cuda().float(), self.test_data.cuda().float())
        
        return test_res
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
