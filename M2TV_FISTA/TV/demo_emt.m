clc;close all; clear;

addpath('.\HNO'); addpath('.\data');

% load J (sensitivity matrix), V (measurement matrix), 
% and mask (mask matrix of the round sensing area).
% NOTE: J is normalized in rows;
%       V is subtracted with background signal and normalized.
load('emt.mat'); 

% load the mask image of the ground truth phantom position.
load('ref_img.mat');

% set freq (1,2,3,4)
freq = 4; Vs = V(:,freq);

% 'iso'--> isotropic TV
% 'l1' --> l1-based, anisotropic TV
pars.tv = 'iso';
pars.MAXITER = 200;
X_res = tv_fista(J,Vs,mask,1e-7,-Inf,Inf,pars);

load('.\data\s11.mat');
DispRecos(mask * X_res(:), s11, 64,'linear',0);title('Reconstructed');
DispRecos(mask * ref_img(:), s11, 64,'linear',0);title('Ground Truth Location');
p = psnr(X_res, ref_img); sim = ssim(X_res, ref_img);
