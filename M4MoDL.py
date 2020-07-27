# -*- coding: utf-8 -*-
"""
Created on July 15, 2020

Model-based deep learning method 10.1109/TMI.2018.2865356
iterate:
(1) z_k = D_w (x_k)     denoise
(2) x_{k+1} = (A^T * A + \lambda I)^{-1} ( A^{T} * b + \lambda z_k )    data consistency

@author: XIANG
"""


import torch 
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import radon, iradon

class dw(nn.Module):
    def __init__(self, n_layers=5, in_chl=1, out_chl=1, features=64):
        super(dw, self).__init__()
        self.n_layers = n_layers
        self.in_chl = in_chl
        self.out_chl = out_chl
        self.features = features

        for it in range(self.n_layers):
            block =[]

            if it == 0:
                block.append(nn.Conv2d(in_chl, self.features, kernel_size=3, padding=1, bias=False))
                #block.append(nn.BatchNorm2d(self.features))
                block.append(nn.ReLU(inplace=True))
                mod_block = nn.ModuleList(block)
                setattr(self, 'mod_block_iter_{}'.format(it), mod_block)
            elif it < (self.n_layers-1):
                block.append(nn.Conv2d(self.features, self.features, kernel_size=3, padding=1, bias=False))
                #block.append(nn.BatchNorm2d(self.features))
                block.append(nn.ReLU(inplace=True))
                mod_block = nn.ModuleList(block)
                setattr(self, 'mod_block_iter_{}'.format(it), mod_block)
            else:
                block.append(nn.Conv2d(self.features, out_chl, kernel_size=3, padding=1, bias=False))
                #block.append(nn.BatchNorm2d(out_chl))
                block.append(nn.ReLU())
                mod_block = nn.ModuleList(block)
                setattr(self, 'mod_block_iter_{}'.format(it), mod_block)


    def forward(self, x):
        x_in = x
        for it in range(self.n_layers):
            mod_block = getattr(self, 'mod_block_iter_{}'.format(it))
            for i, mod in enumerate(mod_block):
                x = mod(x)
        x = x + x_in
        return x

class myAtA(object):
    def __init__(self, theta, lam):
        self.theta = theta
        self.lam = lam.cpu().detach().numpy()

    def myAtA(self, img):
        img = img.cpu().detach().numpy()
        Aimg = radon(img, theta=self.theta, circle=True)
        AtAimg = iradon(Aimg, theta=self.theta)
        AtA = AtAimg  + self.lam * img
        AtA = torch.from_numpy(AtA).cuda()
        return AtA

def myCG(A, b):
    """
    A is a class object.
    """
    x = torch.zeros_like(b)
    r = b - A.myAtA(x)
    p = r
    rsold = r*r
    
    for i in range(5):
        Ap = A.myAtA(p)
        pAp = (p*Ap)
        alpha = torch.norm(rsold) / torch.norm(pAp)
        x = x + (alpha * p)
        r = r - (alpha * Ap)
        rsnew = r*r
        if torch.norm(rsnew) < 1e-10:
            break
        beta = torch.norm(rsnew) / torch.norm(rsold)
        p = r + beta * p
        rsold = rsnew
    return x


# Inherit from Function
class dcFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, zk, b, lam, theta):
        ctx.save_for_backward(lam)
        ctx.constant = theta

        Aobj = myAtA(theta, lam)
        xn = torch.zeros_like(zk)
        batch_size = zk.shape[0]
        for i in range(batch_size):
            z_in = zk[i].squeeze()
            b_in = b[i].squeeze()
            b_in = b_in.cpu().detach().numpy()
            
            rhs = iradon(b_in, theta=theta)
            rhs = torch.from_numpy(rhs)
            rhs = rhs.cuda() + lam * z_in
            xn[i] = myCG(Aobj, rhs)

        return xn

    @staticmethod
    def backward(ctx, grad_xk):
        lam = ctx.saved_tensors[0]
        theta = ctx.constant

        Aobj = myAtA(theta, lam)
        grad_zk = torch.zeros_like(grad_xk)
        batch_size = grad_xk.shape[0]
        for i in range(batch_size):
            gradx_in = grad_xk[i].squeeze()
            
            grad_zk[i] = myCG(Aobj, gradx_in)
                
        return grad_zk, None, None, None


class dc(nn.Module):

    def __init__(self, theta):
        super(dc, self).__init__()
        self.theta = theta

        # learnable regularization parameter lambda
        # self.lam = nn.Parameter(torch.Tensor(1))
        # self.lam.data.uniform_(0.05, 0.1)
        # self.Sp = nn.Softplus()

        self.lam = torch.Tensor([0.01]).cuda()

    def forward(self, zk, b):

        # use manual geradients
        return dcFunction.apply(zk, b, self.lam, self.theta)


class MoDL(nn.Module):
    def __init__(self, niter,theta):
        super(MoDL, self).__init__()

        self.niter = niter
        self.dw = dw(n_layers=5, in_chl=1, out_chl=1, features=64)
        self.dc = dc(theta)


    def forward(self, x0, b):

        xk = x0
        for _ in range(self.niter):
            zk = self.dw(xk)     # denoiser
            xk = self.dc(zk, b)  # data consistency
        return xk

