% This script demonstrates some basics of how to generate realistic
% test images in MATLAB.

% Reference: See Chapter 4, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.

% Switch "echo" on to display comment lines.
  echo on

% First read an image whose size is larger than that of the final one.

  Xbig = double(imread('iograyBorder.tif'));

% Create the PSF array for the desired blurring (here: Gaussian blurring).
  
  [P, center] = psfGauss([512,512], 6);

% Create a PSF array whose size matches the large image.

  Pbig = padPSF(P, size(Xbig));

% Compute the blurred large image by a convolution of the sharp
% large image with the corresponding PSF.

  Sbig = fft2(circshift(Pbig, 1-center));  % Eigenvalue of big PSF.
  Bbig = real(ifft2(Sbig .* fft2(Xbig)));  % Blurred large image.

% Extract the central parts of the large images.  

  X = Xbig(51:562,51:562);
  B = Bbig(51:562,51:562);

% Add white Gaussian noise E the the blurred image, scaled such that
% || e ||_2 / || A x ||_2 = 0.01.

  E = randn(size(B));
  E = E / norm(E,'fro');
  B = B + 0.01*norm(B,'fro')*E;
  
  echo off