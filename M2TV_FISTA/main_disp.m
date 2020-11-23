clc;close all; clear;
%% 

addpath('.\TV'); 
addpath('.\data');
addpath('.\npy2matlab');
J = csvread('Jmat.csv');
mask = csvread('mask.csv');

V = readNPY('val_data.npy');
Vs = squeeze(V(15,1,:));


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
len = 1000;
result = zeros(len, 64, 64);
tic

for i=1:7
a = log(1+exp(0.2598*i-0.017));
b = log(1+exp(0.2598-0.017));
(a-b)/a
end


% One-step reconstruction
x_onestep_tv = (Phi'*Phi +  1e-5* Lap)\Phi'* Vs; % TV reg.

% Display GT
%     DispRecos(mask * ref_img(:), 64,'linear',0);title('Ground Truth Location');
DispRecos(x_onestep_tv, 64,'linear',0);title(['TV Reg #']);

% Iterative Total Varaition using FISTA
% 'iso'--> isotropic TV
% 'l1' --> l1-based, anisotropic TV
pars.tv = 'iso';
pars.MAXITER = 200;
pars.fig = 1;
X_fista_tv = tv_fista(Phi,Vs,mask,1e-4,-Inf,Inf,pars);
X_fista_tv = flip(X_fista_tv, 1);
DispRecos(mask * X_fista_tv(:), 64,'linear',0);title(['Iterative TV #']);

writeNPY(X_fista_tv, '.\results\row2_tv.npy')
