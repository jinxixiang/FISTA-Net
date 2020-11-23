
from loader import DataSplit
import numpy as np
from helpers import show_image_matrix
import torch 
from M1LapReg import callLapReg, MatMask
from M3FBPConv import FBPConv
from M4ISTANet import ISTANet
from M5FISTANet import FISTANet
import argparse
from solver import Solver
import os
from os.path import dirname, join as pjoin
from metric import compute_measure

# =============================================================================
# Load dataset
# some parameters of the dataset
batch_size = 64
validation_split = 0.2

# load input and target dataset; test_loader, train_loader, val_loader
root_dir = "../data/EITData/CircleCases_MultiLevel"
train_loader, val_loader, test_loader = DataSplit(root_dir=root_dir,
                                                  batch_size=batch_size,
                                                  validation_split=validation_split)


# =============================================================================
# 
# Get 100 batch of test data and validate data
for i, (y_v, images_v) in enumerate(test_loader):
    if i==0:
        test_images = images_v
        test_data = y_v
    elif i==2:
        break
    else:
        test_images = torch.cat((test_images, images_v), axis=0)
        test_data = torch.cat((test_data, y_v), axis=0)

# add channel axis; torch tensor format (batch_size, channel, width, height)
test_images = torch.unsqueeze(test_images, 1)  # torch.Size([128, 1, 64, 64])
test_data = torch.unsqueeze(test_data, 1)
test_data = torch.unsqueeze(test_data, 3)      # torch.Size([128, 1, 104, 1])

# print("Shape of test dataset: {}".format(test_images.shape))


# =============================================================================
# Model 1
# Reconstruction with Laplacian Regularization 
print('===========================================')
print('Laplacian Regularization...')
data_dir = "../data/EITData"
x_lap = callLapReg(data_dir=data_dir, y_test=test_data)

results = [test_images, x_lap]
titles = ['Truth', 'Lap. Reg']
dir_name = "./figures"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('Create path : {}'.format(dir_name))
# Evalute reconstructed images with PSNR, SSIM, RMSE.
p_reg, s_reg, m_reg = compute_measure(test_images, x_lap, 1)
print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_reg, s_reg, m_reg))
show_image_matrix(dir_name+"/LapFigs.png", results, titles=titles, indices=slice(0,15))

# =============================================================================
# Model 2
# Total Variation with FISTA (https://sites.google.com/site/amirbeck314/software)
# Digital Object Identifier 10.1109/TIP.2009.2028250



# =============================================================================
# Model 3
# Lap. Reg. + U-net
# Use these parameters to steer the training
print('===========================================')
print('Lap. Reg. + U-net...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FBPConv_mode = 0    # 0, test mode; 1, train mode.
FBPConv = FBPConv().to(device)

print('Total number of parameters FBPConv: ',
      sum(p.numel() for p in FBPConv.parameters()))

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='FBPConv')
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--data_dir', type=str, default=data_dir)
parser.add_argument('--save_path', type=str, default='./models/FBPConv/')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--device',  default=device)
parser.add_argument('--log_interval', type=int, default=20)
parser.add_argument('--test_epoch', type=int, default=30)
args = parser.parse_args()

if args.start_epoch > 0:
    f_trained = pjoin(args.save_path, 'epoch_{}.ckpt'.format(args.start_epoch))
    FBPConv.load_state_dict(torch.load(f_trained))
        
solver = Solver(FBPConv, train_loader, args, test_data)

if FBPConv_mode==1:
    solver.train()
    FBPConv_test = solver.test()
else:
    FBPConv_test = solver.test()

FBPConv_test = FBPConv_test.cpu().double()
fig_name = dir_name + '/FBPConv_' + str(args.test_epoch) + 'epoch.png'
results = [test_images, x_lap, FBPConv_test]
# Evalute reconstructed images with PSNR, SSIM, RMSE.
p_unet, s_unet, m_unet = compute_measure(test_images, FBPConv_test, 1)
print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_unet, s_unet, m_unet))
titles = ['truth', 'Lap. Reg.', 'FBPConv']
show_image_matrix(fig_name, results, titles=titles, indices=slice(0, 15))

# =============================================================================
# Model 4
# ISTA-Net (DOI 10.1109/CVPR.2018.00196)

print('===========================================')
print('ISTA-Net CVPR 2018...')

ISTANet_mode = 0
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# load sensitivity matrix of EIT
J_eit = np.loadtxt(pjoin(data_dir, "Jmat.csv"), delimiter=",", dtype=float)
J_eit = torch.from_numpy(J_eit)
mask = MatMask(64)
#mask = torch.from_numpy(mask)
mask = torch.tensor(mask, dtype=torch.float32, device=device)
J_eit = torch.tensor(J_eit, dtype=torch.float32, device=device)

LayerNo = 6
ISTANet = ISTANet(LayerNo=LayerNo, Phi=J_eit, mask=mask)
ISTANet = ISTANet.to(device)

print('Total number of parameters ISTANet net:',
      sum(p.numel() for p in ISTANet.parameters()))

    
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='ISTANet')
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--data_dir', type=str, default=data_dir)
parser.add_argument('--save_path', type=str, default='./models/ISTANet/')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--device', default=device)
parser.add_argument('--log_interval', type=int, default=20)
parser.add_argument('--test_epoch', type=int, default=30)
args = parser.parse_args()

if args.start_epoch > 0:
    f_trained = pjoin(args.save_path, 'epoch_{}.ckpt'.format(args.start_epoch))
    ISTANet.load_state_dict(torch.load(f_trained))

solver = Solver(ISTANet, train_loader, args, test_data)

if ISTANet_mode==1:
    solver.train()
    ISTANet_test = solver.test()
else:
    ISTANet_test = solver.test()

ISTANet_test = ISTANet_test.cpu().double()
fig_name = dir_name + '/ISTANet_' + str(args.test_epoch)  + 'epoch.png'
results = [test_images, x_lap, FBPConv_test, ISTANet_test]
# Evalute reconstructed images with PSNR, SSIM, RMSE.
p_ISTANet, s_ISTANet, m_ISTANet = compute_measure(test_images, ISTANet_test, 1)
print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_ISTANet, s_ISTANet, m_ISTANet))
titles = ['truth', 'Lap. Reg.',  'FBPConv', 'ISTANet']
show_image_matrix(fig_name, results, titles=titles, indices=slice(0, 15))


# =============================================================================
# Model 5
# FISTA-Net: Proposed method

print('===========================================')
print('FISTA-Net...')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load sensitivity matrix of EIT; load mask matrix, dim from 3228 to 4096
J_eit = np.loadtxt(pjoin(data_dir, "Jmat.csv"), delimiter=",", dtype=float)
J_eit = torch.from_numpy(J_eit)
J_eit = torch.tensor(J_eit, dtype=torch.float32, device=device)

# row and column difference matrix for TV

DM = np.loadtxt(pjoin(data_dir, "DM.csv"), delimiter=",", dtype=float)
DMts = torch.from_numpy(DM) 
mask = MatMask(64)
mask = torch.from_numpy(mask)
mask = torch.tensor(mask, dtype=torch.float32, device=device)
DMts = torch.tensor(DMts, dtype=torch.float32, device=device)

fista_net_mode = 0    # 0, test mode; 1, train mode.
fista_net = FISTANet(7, J_eit, DMts, mask)
fista_net = fista_net.to(device)

print('Total number of parameters fista net:',
      sum(p.numel() for p in fista_net.parameters()))

# define arguments of fista_net
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='FISTANet')
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--data_dir', type=str, default=data_dir)
parser.add_argument('--save_path', type=str, default='./models/FISTANet/')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--device', default=device)
parser.add_argument('--log_interval', type=int, default=20)
parser.add_argument('--test_epoch', type=int, default=30)
args = parser.parse_args()

if args.start_epoch > 0:
    f_trained = pjoin(args.save_path, 'epoch_{}.ckpt'.format(args.start_epoch))
    fista_net.load_state_dict(torch.load(f_trained))

solver = Solver(fista_net, train_loader, args, test_data)

if fista_net_mode == 1:
    solver.train()
    fista_net_test = solver.test()
else:
    fista_net_test = solver.test()

fista_net_test = fista_net_test.cpu().double()
fig_name = dir_name + '/fista_net_' + str(args.test_epoch) + 'epoch.png'
results = [test_images, x_lap, FBPConv_test,  ISTANet_test, fista_net_test]
# Evalute reconstructed images with PSNR, SSIM, RMSE.
p_fista, s_fista, m_fista = compute_measure(test_images, fista_net_test, 1)
print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_fista, s_fista, m_fista))
titles = ['truth', 'Lap. Reg.','FBPConv', 'ISTANet', 'fista_net']
show_image_matrix(fig_name, results, titles=titles, indices=slice(0, 15))