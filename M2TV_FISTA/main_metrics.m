clc;close all; clear;
%% 

addpath('./TV'); 
addpath('./data');
addpath('./npy2matlab');
J = csvread('Jmat.csv');
mask = csvread('mask.csv');

data_dir = '..\EMTData\CircleCases_MultiLevel\';
V = readNPY('test_data.npy');
Img1 = readNPY('test_image.npy');
Img1 = flip(Img1, 2);
% One-step Inversion with Regularization

% compute difference operators
dr_filt = [1;-1];
dc_filt = [1,-1];
Dr = convmtx2(dr_filt, 64,64) * mask';
Dc = convmtx2(dc_filt, 64,64) * mask';
DtDr = Dr'*Dr;
DtDc = Dc'*Dc;
Lap = (DtDc + DtDr);

Phi = J;
ssim_ = 0;
psnr_ = 0;
len = size(Img1, 1);
result = zeros(len, 64, 64);
tic
for i = 1:len  
    % load one sample
    Vs = V(i,:)'; ref_img = Img1(i,:);
%     Vs = csvread('./data/row2_data.csv');
    % One-step reconstruction
%     x_onestep_tv = (Phi'*Phi +  1e-5* Lap)\Phi'* Vs; % TV reg.

    % Display GT
%     DispRecos(mask * ref_img(:), 64,'linear',0);title('Ground Truth Location');
%     DispRecos(x_onestep_tv, 64,'linear',0);title(['TV Reg #' num2str(i)]);
    
    % Iterative Total Varaition using FISTA
    % 'iso'--> isotropic TV
    % 'l1' --> l1-based, anisotropic TV
    pars.tv = 'iso';
    pars.MAXITER = 10;
    pars.fig = 0;
    X_fista_tv = tv_fista(Phi,Vs,mask,1e-4,-Inf,Inf,pars);
    result(i,:,:) = X_fista_tv;
    img_gt = reshape(ref_img,64,64); 
    ssim_ = ssim_ + ssim(X_fista_tv, img_gt);
    psnr_ = psnr_ + psnr(X_fista_tv, img_gt, 1);
    DispRecos(mask * X_fista_tv(:), 64,'linear',0);title(['Iterative TV #' num2str(i)]);
%     writeNPY(X_fista_tv, './results/iteration_row2/iter10.npy');
end
toc
ssim_ = ssim_ / len;
psnr_ = psnr_ / len;
save('./results/reconstructed_image_test.mat','result');
save('./results/psnr_test.mat','psnr_');
save('./results/ssim_test.mat','ssim_');
