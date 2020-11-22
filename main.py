
import numpy as np
import torch 
from skimage.transform import radon, iradon
import argparse
import os
from os.path import dirname, join as pjoin

from loader import DataSplit
from helpers import show_image_matrix, show_batch
from solver import Solver
from metric import compute_measure

from M3FBPConv import FBPConv
from M4ISTANet import ISTANet
from M5FISTANet import FISTANet

# =============================================================================
# Load dataset
# load input and target dataset; test_loader, train_loader, val_loader
data_dir = '/media/ps/D/LDCT/npy_img'
ds_factor = 12
batch_size = 1
train_loader, val_loader, test_loader = DataSplit(test_patient='L096', batch_size=batch_size, validation_split=0.2, 
                                                  ds_factor=ds_factor, saved_path=data_dir,  transform=None)


print("Size of train_loader: ", len(train_loader))
print("Size of val_loader: ", len(val_loader))
print("Size of test_loader: ", len(test_loader))


# =============================================================================
# 
# Get one batch of test data and validate data
for i, (y_v, images_v) in enumerate(test_loader):
    if i==0:
        test_images = images_v
        test_data = y_v
    # elif i==1:
    #     break
    # else:
    #     test_images = torch.cat((test_images, images_v), axis=0)
    #     test_data = torch.cat((test_data, y_v), axis=0)

# add channel axis; torch tensor format (batch_size, channel, width, height)
test_images = torch.unsqueeze(test_images, 1)   # torch.Size([batch_size, 1, 512, 512])
test_data = torch.unsqueeze(test_data, 1)       # torch.Size([batch_size, 1, 512, 720/ds])

print("Size of test dataset: {}".format(test_images.shape))
print("Size of measurements: {}".format(test_data.shape))

# appoint test dataset
# test_images = np.load('test_images.npy') 
# test_data = np.load('test_data.npy') 
# test_images = torch.from_numpy(test_images)
# test_data = torch.from_numpy(test_data)


dir_name = "./figures"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('Create path : {}'.format(dir_name))


num_display = test_images.shape[0]

# =============================================================================
# Model 1
# Filtered Back Projection with iradon transform
print('===========================================')
print('FBP...')
rotView = test_data.shape[3]
theta = np.linspace(0.0, 180.0, rotView, endpoint=False)

X_fbp = torch.zeros_like(test_images)

for i in range(num_display):
    sino = test_data[i].squeeze()
    X0 = iradon(sino, theta=theta)
    X_fbp[i] = torch.from_numpy(X0)

results = [test_images, X_fbp]
titles = ['Truth', 'LBP']
show_image_matrix(dir_name+"/LBP.png", results, titles=titles, indices=slice(0, num_display))

# Evalute reconstructed images with PSNR, SSIM, RMSE.
p_reg, s_reg, m_reg = compute_measure(test_images, X_fbp, 1)
print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_reg, s_reg, m_reg))

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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

FBPConv_mode = 0    # 0, test mode; 1, train mode.
FBPConv = FBPConv(in_channels = 1, out_channels = 1, features = 16)
FBPConv = FBPConv.to(device)
 

print('Total number of parameters FBPConv: ',
      sum(p.numel() for p in FBPConv.parameters()))

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='FBPConv')
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--data_dir', type=str, default=data_dir)
parser.add_argument('--save_path', type=str, default='./models/FBPConv/')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--device', default=device)
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--theta', type=float, default=theta)
args = parser.parse_args()

if args.start_epoch > 0:
    f_trained = pjoin(args.save_path, 'epoch_{}.ckpt'.format(args.start_epoch))
    FBPConv.load_state_dict(torch.load(f_trained))
        
solver = Solver(FBPConv, train_loader, args, test_data, test_images)

if FBPConv_mode==1:
    solver.train()
    FBPConv_test = solver.test()
else:
    FBPConv_test = solver.test()

FBPConv_test = FBPConv_test.cpu().double()
fig_name = dir_name + '/FBPConv_' + str(args.test_epoch) + 'epoch.png'
results = [test_images, X_fbp, FBPConv_test]
titles = ['truth', 'FBP', 'FBPConv']
show_image_matrix(fig_name, results, titles=titles, indices=slice(0, num_display))

# Evalute reconstructed images with PSNR, SSIM, RMSE.
p_reg, s_reg, m_reg = compute_measure(test_images, FBPConv_test, 1)
print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_reg, s_reg, m_reg))

# =============================================================================
# Model 4
# ISTANet CVPR 2018

print('===========================================')
print('ISTANet CVPR 2018...')

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

ISTANet_mode = 0
LayerNo = 5
ISTANet = ISTANet(LayerNo=LayerNo, theta=theta)
ISTANet = ISTANet.to(device)

print('Total number of parameters ISTANet:',
      sum(p.numel() for p in ISTANet.parameters()))

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='ISTANet')
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--data_dir', type=str, default=data_dir)
parser.add_argument('--save_path', type=str, default='./models/ISTANet/')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--device', default=device)
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--theta', type=float, default=theta)
args = parser.parse_args()

if args.start_epoch > 0:
    f_trained = pjoin(args.save_path, 'epoch_{}.ckpt'.format(args.start_epoch))
    ISTANet.load_state_dict(torch.load(f_trained))

solver = Solver(ISTANet, train_loader, args, test_data, test_images)

if ISTANet_mode==1:
    solver.train()
    ISTANet_test = solver.test()
else:
    ISTANet_test = solver.test()

ISTANet_test = ISTANet_test.cpu().double()
fig_name = dir_name + '/ISTANet_' + str(args.test_epoch)  + 'epoch.png'
results = [test_images, X_fbp, FBPConv_test, ISTANet_test]
titles = ['truth', 'LBP',  'denoise net', 'ISTANet']
show_image_matrix(fig_name, results, titles=titles, indices=slice(0, num_display))

# Evalute reconstructed images with PSNR, SSIM, RMSE.
p_reg, s_reg, m_reg = compute_measure(test_images, ISTANet_test, 1)
print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_reg, s_reg, m_reg))

# =============================================================================
# Model 5
# FISTA-Net: Proposed method

print('===========================================')
print('FISTA-Net...')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

FISTANet_mode = 1    # 0, test mode; 1, train mode.
FISTANet = FISTANet(6, theta)
FISTANet = FISTANet.to(device)

print('Total number of parameters fista net:',
      sum(p.numel() for p in FISTANet.parameters()))

# define arguments of FISTANet
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='FISTANet')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--data_dir', type=str, default=data_dir)
parser.add_argument('--save_path', type=str, default='./models/FISTANet/')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--device', default=device)
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--test_epoch', type=int, default=10)
parser.add_argument('--theta', type=float, default=theta)
args = parser.parse_args()

if args.start_epoch > 0:
    f_trained = pjoin(args.save_path, 'epoch_{}.ckpt'.format(args.start_epoch))
    FISTANet.load_state_dict(torch.load(f_trained))

solver = Solver(FISTANet, train_loader, args, test_data, test_images)

if FISTANet_mode == 1:
    solver.train()
    FISTANet_test = solver.test()
else:
    FISTANet_test = solver.test()

FISTANet_test = FISTANet_test.cpu().double()
fig_name = dir_name + '/FISTANet_' + str(args.test_epoch) + 'epoch.png'
results = [test_images, X_fbp, FBPConv_test, ISTANet_test, FISTANet_test]
titles = ['truth', 'FBP', 'FBPConv', 'ISTANet', 'FISTANet']
show_image_matrix(fig_name, results, titles=titles, indices=slice(0, num_display))

# Evalute reconstructed images with PSNR, SSIM, RMSE.
p_reg, s_reg, m_reg = compute_measure(test_images, FISTANet_test, 1)
print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_reg, s_reg, m_reg))
