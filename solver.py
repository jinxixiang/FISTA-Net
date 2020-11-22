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
        assert args.model_name in ['FBPConv', 'ISTANet', 'FISTANet']

        self.model_name = args.model_name
        self.model = model
        self.data_loader = data_loader
        self.data_dir = args.data_dir
        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.lr = args.lr


        if self.model_name == 'FISTANet':
            # set different lr for regularization weights and network weights
            self.optimizer = optim.Adam([
            {'params': self.model.module.fcs.parameters()}, 
            {'params': self.model.module.w_theta, 'lr': 1e-4},
            {'params': self.model.module.b_theta, 'lr': 1e-1},
            {'params': self.model.module.w_mu, 'lr': 1e-4},
            {'params': self.model.module.b_mu, 'lr': 1e-4},
            {'params': self.model.module.w_rho, 'lr': 1e-4},
            {'params': self.model.module.b_rho, 'lr': 1e-4}], 
            lr=self.lr, weight_decay=0.001)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), self.lr, weight_decay=0.001)
        
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1) # step-wise
        
        self.theta = args.theta
        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu
        self.device = args.device
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

                # initial guess x0
                X_fbp = torch.tensor(X_fbp, dtype=torch.float32, device=self.device)         
                # ground truth image
                y_target = torch.tensor(y_target, dtype=torch.float32, device=self.device)   
                
                # predict and compute losses
                if self.model_name == 'ISTANet':
                    [pred, loss_sym] = self.model(X_fbp, b_in)

                    # Compute loss, data consistency and regularizer constraints
                    loss_discrepancy = self.train_loss(pred, y_target) + l1_loss(pred, y_target, 0.1)
                    loss_constraint = 0
                    for k, _ in enumerate(loss_sym, 0):
                        loss_constraint += torch.mean(torch.pow(loss_sym[k], 2))

                    loss = loss_discrepancy + 0.01 * loss_constraint

                if self.model_name == 'FISTANet':
                    [pred, loss_layers_sym, encoder_st] = self.model(X_fbp, b_in)

                    # Compute loss, data consistency and regularizer constraints
                    loss_discrepancy = self.train_loss(pred, y_target) + l1_loss(pred, y_target, 0.1)
                    loss_constraint = 0
                    for k, _ in enumerate(loss_layers_sym, 0):
                        loss_constraint += torch.mean(torch.pow(loss_layers_sym[k], 2))

                    encoder_constraint = 0
                    for k, _ in enumerate(encoder_st, 0):
                        encoder_constraint += torch.mean(torch.abs(encoder_st[k]))

                    # loss = loss_discrepancy + gamma * loss_constraint
                    loss = loss_discrepancy +  0.01 * loss_constraint + 0.001 * encoder_constraint
                if self.model_name == 'FBPConv':
                    pred = self.model(X_fbp)
                    loss = self.train_loss(pred, y_target) + l1_loss(pred, y_target, 0.1)
                
                self.model.zero_grad()
                self.optimizer.zero_grad()
                
                # backpropagate the gradients
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
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
            
            X_fbp = torch.tensor(X_fbp, dtype=torch.float32, device=self.device)
            self.test_data = torch.tensor(self.test_data, dtype=torch.float32, device=self.device)

            if self.model_name == "FBPConv":
                test_res = self.model(X_fbp)

            if self.model_name == "FISTANet":
                [test_res, _, _] = self.model(X_fbp, self.test_data)

            if self.model_name == "ISTANet":
                [test_res, _] = self.model(X_fbp, self.test_data)

        
        return test_res
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
