% HNO Functions. 
% Version 1.1  27-April-07. 
% Copyright (c) 2006 by P. C. Hansen, J. G. Nagy, and D. P. O'Leary.
%
% Requires Matlab version 6.5 or later versions.
%    
% These functions accompany the book
%   Deblurring Images - Matrices, Spectra, and Filtering
%   P. C. Hansen, J. G. Nagy, and D. P. O'Leary
%   SIAM, Philidelphia, 2006.
%  
% Demonstration. 
%   chapter2demo - Demonstrates how to read, display and manipulate images. 
%   chapter4demo - Demonstrates how to generate realistic test images. 
%  
% Deblurring functions. 
%   gtik_fft - Generalized Tikhonov image deblurring using the FFT algorithm.
%   tik_dct  - Tikhonov image deblurring using the DCT algorithm
%   tik_fft  - Tikhonov image deblurring using the FFT algorithm
%   tik_sep  - Tikhonov image deblurring using the Kronecker decomposition
%   tsvd_dct - Truncated SVD image deblurring using the DCT algorithm
%   tsvd_fft - Truncated SVD image deblurring using the FFT algorithm
%   tsvd_sep - Truncated SVD image deblurring using Kronecker decomposition
%    
% Generalized cross validation.
%   gcv_gtik - GCV parameter choice method for gtik_fft deblurring function
%   gcv_tik  - GCV parameter choice method for Tikhonov image deblurring
%   gcv_tsvd - GCV parameter choice method for TSVD image deblurring
%    
% Point spread functions.
%   psfDefocus - Array with point spread function for out-of-focus blur
%   psfGauss   - Array with point spread function for Gaussian blur
%    
% Auxiliary functions.
%   dctshift   - Create array containing the first column of a blurring matrix
%   kronDecomp - Kronecker product decomposition of PSF array
%   padPSF     - Pad a PSF array with zeros to make it bigger
%    
% Our DCT functions (for users without access to the Signal Processing and
% Image Processing toolboxes).
%   dcts   - Model implementation of discrete cosine transform
%   dcts2  - Model implementation of 2-D discrete cosine transform
%   idcts  - Model implementation of inverse discrete cosine transform
%   idcts2 - Model implementation of 2-D inverse discrete cosine transform