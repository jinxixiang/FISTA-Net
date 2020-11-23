# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:47:30 2020

One step Laplacian Regularization:
    
    \hat{x} = ( J^T J + \lambda R )^{-1}J^Ty

where, R is a Laplacian matrix. 

input   y:  (104, 1) measurements
output  x:  (64, 64) images.     

@author: XIANG
"""
import torch 
import numpy as np
from os.path import dirname, join as pjoin


def LapReg(J, y, lda, Lap):
    """
    Laplacian Inversion method.
    Input:
        J: sensitivity matrix, 104*3228;
        y: measurement vector, 104*1;
        lda: coeficient;
        Lap: Regularization Matrix, 3228*3228;
    Output:
        x_est: conductivity distribution
    """
    Jt = J.t()
    x_est = torch.inverse(torch.add(torch.mm(Jt, J), lda * Lap))
    x_est = torch.mm(x_est, Jt)    # here, X_est 3228x104,
    x_est = torch.mm(x_est, y)     # here, X_est 3228*1, y 104*1
    
    
    return x_est

def MatMask(pnum):
    """
    Create mask matrix to convert 3228 1d vector to 64*64 2d image.
    Dimension: (3228, 4096)
    """
    # generate coordinates in an unit circle
    xcor = np.arange(-1+1/pnum, 1+1/pnum, 2/pnum)    
    ycor = np.arange(1-1/pnum, -1-1/pnum, -2/pnum)

    Msk = np.zeros((3228, 4096))
    Mat_id = np.arange(4096).reshape(pnum, pnum)
    
    Mat_id = Mat_id.T
    Mat_id = np.fliplr(Mat_id)

    k = 0
    for j in range(pnum):
        for i in range(pnum):
            if (xcor[i]*xcor[i]+ycor[j]* ycor[j] <= 1):
                Msk[k, Mat_id[j,i]] = 1
                k = k +1
            else:
                k = k
    return Msk

def Convert2dImg(xest, pnum):
    """
    Convert 1d results to 2d image
    Input:
        xest, 1d numpy vector 3228*1 or tensor (3228, batch_size)
        pnum, mesh grid number;
        tensorout,  tensorout=0, return numpy array (pnum*sr, pnum*sr);
                    tensorout=1, return tensor (batch_size, pnum, pnum).
    """    
    msk_np = MatMask(pnum)
    msk_tensor = torch.tensor(msk_np)               # dim: (3228, 4096)
    msk_tensor = torch.transpose(msk_tensor, 0, 1)
    result_tensor_1d = torch.mm(msk_tensor, xest)  # (4096, 3228) * (3228, batch_size)
    result_tensor_2d = result_tensor_1d.view(pnum, pnum, -1)
    return result_tensor_2d.permute(2, 0, 1)

def callLapReg(data_dir, y_test):
    """
    data_dir: data directory contains the sensitivity matrix and regularization matrix;
    y_test: one batch of measuremnet test data.
    """
    LapMat = np.loadtxt(pjoin(data_dir, "Lapmat.csv"), delimiter=",", dtype=float)
    J = np.loadtxt(pjoin(data_dir, "Jmat.csv"), delimiter=",", dtype=float)
    
    Lts = torch.from_numpy(LapMat)    # Laplacian Regularization matrix in tensor 
    Jts = torch.from_numpy(J)         # Sensitivity matrix in tensor
    
    # convert data format from (batch_size, channel, vector_row, vector_col) = (128, 1, 104, 1)
    # to (vector_row, batch_size) = (104, 128)
    test_data_lap = torch.squeeze(y_test, 1)
    test_data_lap = torch.squeeze(test_data_lap, 2)
    test_data_lap = test_data_lap.t()
    
    # solve with Laplacian Regularization
    x_est = LapReg(Jts, test_data_lap, 0.001, Lts)               # 1d tensor results
    
    
    x_lap = Convert2dImg(x_est, pnum=64)    # convert 1d to 2d 
    x_lap = torch.unsqueeze(x_lap, 1)       # (128, 1, 64, 64)
    return x_lap
