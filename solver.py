# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:26:29 2020

@author: XIANG
"""
import torch.optim as optim
from M1LapReg import callLapReg
import torch
import torch.nn as nn
from os.path import dirname, join as pjoin
from collections import OrderedDict
import time
import numpy as np
from helpers import test_rescale, show_image_matrix
#from torch.utils.tensorboard import SummaryWriter
import os

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
            {'params': self.model.fcs.parameters()}, 
            {'params': self.model.w_theta, 'lr': 0.001},
            {'params': self.model.b_theta, 'lr': 0.001},
            {'params': self.model.w_mu, 'lr': 0.001},
            {'params': self.model.b_mu, 'lr': 0.001},
            {'params': self.model.w_rho, 'lr': 0.001},
            {'params': self.model.b_rho, 'lr': 0.001}], 
            lr=self.lr, weight_decay=0.001)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.001)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9) # step-wise


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
        #writer = SummaryWriter('runs/'+self.model_name)
        

        for epoch in range(1 + self.start_epoch, self.num_epochs + 1 + self.start_epoch):
            self.model.train(True)
                
            for batch_idx, (x_in, y_target) in enumerate(self.data_loader):
                
                # measured vector (104*1); add channels
                x_in = torch.unsqueeze(x_in, 1)  
                x_in = torch.unsqueeze(x_in, 3)
                
                # initial image from one-step inversion
                x_img =  callLapReg(data_dir=self.data_dir, y_test=x_in) 
                
                # target image (64*64)
                y_target = torch.unsqueeze(y_target, 1)

                if epoch == 1 and batch_idx == 1:
                    show_image_matrix('./figures/X0.png', [x_img, y_target], 
                        titles=['X0 FBP', 'y_target'], indices=slice(0,30)) 


                x_img = torch.tensor(x_img, dtype=torch.float32, device=self.device)
                x_in = torch.tensor(x_in, dtype=torch.float32, device=self.device)
                y_target = torch.tensor(y_target, dtype=torch.float32, device=self.device)


                if self.model_name == 'FBPConv':

                    pred = self.model(x_img)                     
                    loss = self.train_loss(pred, y_target) + l1_loss(pred, y_target, 0.1)

                # predict and compute losses
                if self.model_name == 'ISTANet':
                    [pred, loss_sym] = self.model(x_img, x_in)
                    loss_discrepancy = self.train_loss(pred, y_target) + l1_loss(pred, y_target, 0.1)
                    loss_constraint = 0
                    for k, _ in enumerate(loss_sym, 0):
                        loss_constraint += torch.mean(torch.pow(loss_sym[k], 2))

                    loss = loss_discrepancy + 0.01 * loss_constraint

                if self.model_name == 'FISTANet':
                    [pred, loss_layers_sym, loss_st] = self.model(x_img, x_in)

                    # Compute loss, data consistency and regularizer constraints
                    loss_discrepancy = self.train_loss(pred, y_target) + l1_loss(pred, y_target, 0.1)
                    loss_constraint = 0
                    for k, _ in enumerate(loss_layers_sym, 0):
                        loss_constraint += torch.mean(torch.pow(loss_layers_sym[k], 2))
                    
                    sparsity_constraint = 0
                    for k, _ in enumerate(loss_st, 0):
                        sparsity_constraint += torch.mean(torch.abs(loss_st[k]))
                    
                    # loss = loss_discrepancy + gamma * loss_constraint
                    loss = loss_discrepancy +  0.01 * loss_constraint + 0.001 * sparsity_constraint

                
                self.model.zero_grad()
                self.optimizer.zero_grad()
                
                # backpropagate the gradients
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                train_losses.append(loss.item())
                

                # print processes 
                if batch_idx % self.log_interval == 0:
                    #writer.add_scalar('training loss', loss.data, epoch * len(self.data_loader) + batch_idx)

                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}\t TIME:{:.1f}s'
                          ''.format(epoch, batch_idx * len(x_in),
                                    len(self.data_loader.dataset),
                                    100. * batch_idx / len(self.data_loader),
                                    loss.data,
                                    time.time() - start_time))  

                    # print weight values of model
                    if self.model_name == 'FISTANet':

                        print("Threshold value w: {}".format(self.model.w_theta))
                        print("Threshold value b: {}".format(self.model.b_theta))
                        print("Gradient step w: {}".format(self.model.w_mu))
                        print("Gradient step b: {}".format(self.model.b_mu))
                        print("Two step update w: {}".format(self.model.w_rho))
                        print("Two step update b: {}".format(self.model.b_rho))
  
            # save model
            if epoch % 1 == 0:

                self.save_model(epoch)
                np.save(pjoin(self.save_path, 'loss_{}_epoch.npy'.format(epoch)), np.array(train_losses))
        
    def test(self):
        self.load_model(self.test_epoch)
        self.model.eval()
        
        with torch.no_grad():
            # Must use the sample test dataset!!!
            x_test = callLapReg(data_dir=self.data_dir, y_test=self.test_data)
            x_test = torch.tensor(x_test, dtype=torch.float32, device=self.device)

            if self.model_name == "ISTANet":
                test_data = torch.tensor(self.test_data, dtype=torch.float32, device=self.device)
                [test_res, _] = self.model(x_test, test_data)

            elif self.model_name == 'FISTANet':
                test_data = torch.tensor(self.test_data, dtype=torch.float32, device=self.device)
                [test_res, _, _] = self.model(x_test, test_data)

            else:
                # other nets needs to do one-step inversion
                test_res = self.model(x_test)

        
        return test_res
        