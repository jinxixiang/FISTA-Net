# -*- coding: utf-8 -*-
"""
Created on June 22, 2020

ISTANet(shared network with 4 conv + ReLU) + regularized hyperparameters softplus(w*x + b). 
The Intention is to make gradient step \mu and thresholding value \theta positive and monotonically decrease.
Replace the Conv+ReLU proximal opeartor with Residual in Residual Dense Block (ESRGAN ECCV 2018). 


@author: XIANG
"""

import torch 
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np 
import os

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=32, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, soft_thr):
        # soft_thr is the beta in Fig.4, ESRGAN 2018.
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = F.relu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = F.relu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * soft_thr + x

# define basic block of FISTA-Net
class  BasicBlock(nn.Module):
    """docstring for  BasicBlock"""

    def __init__(self, in_nc=1, out_nc=1, nf=32, gc=32):
        """
        in_nc:  input number of channels;
        out_nc: output number of channels;
        nf:     number of filters in RDB;
        gc:     growth channel, i.e. intermediate channels.
        """
        super(BasicBlock, self).__init__()
      
        self.Sp = nn.Softplus()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.trunk = ResidualDenseBlock_5C(nf=nf, gc=gc)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

    def forward(self, x, PhiTPhi, PhiTb, LTL, mask, lambda_step, soft_thr):
        
        # convert data format from (batch_size, channel, pnum, pnum) to (circle_num, batch_size)
        pnum = x.size()[2]
        x = x.view(x.size()[0], x.size()[1], pnum**2, -1)   # (batch_size, channel, pnum*pnum, 1)
        x = torch.squeeze(x, 1)
        x = torch.squeeze(x, 2).t()             
        x = mask.mm(x)  
        
        # rk block in the paper
        #x = x - self.lambda_step  * PhiTPhi.mm(x) + self.lambda_step * PhiTb
        # Quadratic TV update
        x = x - self.Sp(lambda_step) * torch.inverse(PhiTPhi + 0.001 * LTL).mm(PhiTPhi.mm(x) - PhiTb + 0.001 * LTL.mm(x))

        # convert (circle_num, batch_size) to (batch_size, channel, pnum, pnum)
        x = torch.mm(mask.t(), x)
        x = x.view(pnum, pnum, -1)
        x = x.unsqueeze(0)
        x_input = x.permute(3, 0, 1, 2)
        
        # Dk block in the paper
        x_D = self.conv_first(x_input)

        # proximal operator, Residual Dense Net
        x = self.trunk(x_D, self.Sp(soft_thr))

        # Gk block in the paper
        x_G = self.conv_last(x)

        # prediction output (skip connection); non-negative output
        x_pred = F.relu(x_input + x_G)

        # dummy symloss here
        symloss = x_D - x_G

        return [x_pred, symloss]


class FISTANet(nn.Module):
    def __init__(self, LayerNo, Phi, L, mask):
        super(FISTANet, self).__init__()
        self.LayerNo = LayerNo
        self.Phi = Phi
        self.L = L
        self.mask =mask
        onelayer = []

        self.bb = BasicBlock()
        for i in range(LayerNo):
            onelayer.append(self.bb)

        self.fcs = nn.ModuleList(onelayer)
        self.fcs.apply(initialize_weights)
        
        # thresholding value
        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([0]))
        # gradient step
        self.w_mu = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu = nn.Parameter(torch.Tensor([0.1]))
        # two-step update weight
        #self.w_rho = nn.Parameter(torch.Tensor([0.5]))
        #self.b_rho = nn.Parameter(torch.Tensor([0]))

    def forward(self, x0, b):
        """
        Phi   : system matrix; default dim 104 * 3228;
        mask  : mask matrix, dim 3228 * 4096
        b     : measured signal vector;
        x0    : initialized x with Laplacian Reg.
        """
        # convert data format from (batch_size, channel, vector_row, vector_col) to (vector_row, batch_size)
        b = torch.squeeze(b, 1)
        b = torch.squeeze(b, 2)
        b = b.t()

        PhiTPhi = self.Phi.t().mm(self.Phi)
        PhiTb = self.Phi.t().mm(b)
        LTL = self.L.t().mm(self.L)
        # initialize the result
        x = x0
        layers_sym = []     # for computing symmetric loss
        for i in range(self.LayerNo):
            theta_ = self.w_theta * i + self.b_theta
            mu_ = self.w_mu * i + self.b_mu
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb, LTL, self.mask, mu_, theta_)
            layers_sym.append(layer_sym)

        return [x, layers_sym]
