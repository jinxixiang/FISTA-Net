
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

from M3DenoiseNet import DenoiseNet
from M4MoDL import MoDL
from M5FISTANet import FISTANet

# =============================================================================
# Load dataset
# load input and target dataset; test_loader, train_loader, val_loader
data_dir = './npy_img/'
ds_factor = 6
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
# test_images = np.load('test_images.npy')
# test_data = np.load('test_data.npy')
# test_images = torch.from_numpy(test_images)
# test_data = torch.from_numpy(test_data)


dir_name = "./figures/"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('Create path : {}'.format(dir_name))

# Display sample data
#show_batch(test_loader, dir_name +'one_sample',0)

num_display = batch_size

# =============================================================================
# Model 1
# Filtered Back Projection with iradon transform
print('===========================================')
print('FBP...')
rotView = test_data.shape[3]
theta = np.linspace(0.0, 180.0, rotView, endpoint=False)

X_fbp = torch.zeros_like(test_images)

for i in range(batch_size):
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
use_cuda = True
denoise_net_mode = 1    # 0, test mode; 1, train mode.
denoise_net = DenoiseNet(in_channels = 1, out_channels = 1, features = 16)

if use_cuda:
    denoise_net = denoise_net.cuda()
 

print('Total number of parameters denoise_net: ',
      sum(p.numel() for p in denoise_net.parameters()))

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='denoise_net')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--data_dir', type=str, default=data_dir)
parser.add_argument('--save_path', type=str, default='./models/denoisenet_120view/')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--use_cuda', type=bool, default=use_cuda)
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--test_epoch', type=int, default=50)
parser.add_argument('--theta', type=float, default=theta)
args = parser.parse_args()

if args.start_epoch > 0:
    f_trained = pjoin(args.save_path, 'epoch_{}.ckpt'.format(args.start_epoch))
    denoise_net.load_state_dict(torch.load(f_trained))
        
solver = Solver(denoise_net, train_loader, args, test_data, test_images)

if denoise_net_mode==1:
    solver.train()
    denoise_net_test = solver.test()
else:
    denoise_net_test = solver.test()

denoise_net_test = denoise_net_test.cpu().double()
fig_name = dir_name + '/denoisenet_' + str(args.test_epoch) + 'epoch.png'
results = [test_images, X_fbp, denoise_net_test]
titles = ['truth', 'FBP', 'denoise net']
show_image_matrix(fig_name, results, titles=titles, indices=slice(0, num_display))

# Evalute reconstructed images with PSNR, SSIM, RMSE.
p_reg, s_reg, m_reg = compute_measure(test_images, denoise_net_test, 1)
print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_reg, s_reg, m_reg))

# =============================================================================
# Model 4
# Model based deep learning method. (Digital Object Identifier 10.1109/TMI.2018.2865356)

print('===========================================')
print('Model based deep learning...')
MoDL_mode = 0

niter = 5
MoDL = MoDL(niter=niter, theta=theta)
if use_cuda:
    MoDL = MoDL.cuda()

print('Total number of parameters MoDL net:',
      sum(p.numel() for p in MoDL.parameters()))

    
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='MoDL')
parser.add_argument('--num_epochs', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--data_dir', type=str, default=data_dir)
parser.add_argument('--save_path', type=str, default='./models/MoDL/')
parser.add_argument('--start_epoch', type=int, default=4)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--use_cuda', type=bool, default=use_cuda)
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--test_epoch', type=int, default=4)
parser.add_argument('--theta', type=float, default=theta)
args = parser.parse_args()

if args.start_epoch > 0:
    f_trained = pjoin(args.save_path, 'epoch_{}.ckpt'.format(args.start_epoch))
    MoDL.load_state_dict(torch.load(f_trained))

solver = Solver(MoDL, train_loader, args, test_data, test_images)

if MoDL_mode==1:
    solver.train()
    MoDL_test = solver.test()
else:
    MoDL_test = solver.test()

MoDL_test = MoDL_test.cpu().double()
fig_name = dir_name + '/MoDL_' + str(args.test_epoch)  + 'epoch.png'
results = [test_images, X_fbp, denoise_net_test, MoDL_test]
titles = ['truth', 'LBP',  'denoise net', 'MoDL']
show_image_matrix(fig_name, results, titles=titles, indices=slice(0, num_display))

# Evalute reconstructed images with PSNR, SSIM, RMSE.
p_reg, s_reg, m_reg = compute_measure(test_images, MoDL_test, 1)
print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_reg, s_reg, m_reg))

# =============================================================================
# Model 5
# FISTA-Net: Proposed method

print('===========================================')
print('FISTA-Net...')
use_cuda = True
fista_net_mode = 1    # 0, test mode; 1, train mode.
fista_net = FISTANet(6, theta)

if use_cuda:
    fista_net = fista_net.cuda()

print('Total number of parameters fista net:',
      sum(p.numel() for p in fista_net.parameters()))

# define arguments of fista_net
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='fista_net')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--data_dir', type=str, default=data_dir)
parser.add_argument('--save_path', type=str, default='./models/fista_net/')
parser.add_argument('--start_epoch', type=int, default=100)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--use_cuda', type=bool, default=use_cuda)
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--test_epoch', type=int, default=150)
parser.add_argument('--theta', type=float, default=theta)
args = parser.parse_args()

if args.start_epoch > 0:
    f_trained = pjoin(args.save_path, 'epoch_{}.ckpt'.format(args.start_epoch))
    fista_net.load_state_dict(torch.load(f_trained))

solver = Solver(fista_net, train_loader, args, test_data, test_images)

if fista_net_mode == 1:
    solver.train()
    fista_net_test = solver.test()
else:
    fista_net_test = solver.test()

fista_net_test = fista_net_test.cpu().double()
fig_name = dir_name + '/fista_net_' + str(args.test_epoch) + 'epoch.png'
results = [test_images, X_fbp, denoise_net_test, fista_net_test]
titles = ['truth', 'FBP','denoise_net', 'fista_net']
show_image_matrix(fig_name, results, titles=titles, indices=slice(0, num_display))

# Evalute reconstructed images with PSNR, SSIM, RMSE.
p_reg, s_reg, m_reg = compute_measure(test_images, fista_net_test, 1)
print('PSNR: {:.5f}\t SSIM: {:.5f} \t RMSE: {:.5f}'.format(p_reg, s_reg, m_reg))
