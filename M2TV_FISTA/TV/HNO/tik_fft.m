function [X, alpha] = tik_fft(B, PSF, center, alpha)
%TIK_FFT Tikhonov image deblurring using the FFT algorithm.
%
%function [X, alpha] = tik_fft(B, PSF, center, alpha)
%
%            X = tik_fft(B, PSF, center);
%            X = tik_fft(B, PSF, center, alpha);
%   [X, alpha] = tik_fft(B, PSF, ...);
%
%  Compute restoration using an FFT-based Tikhonov filter, 
%  with the identity matrix as the regularization operator.
%
%  Input:
%        B  Array containing blurred image.
%      PSF  Array containing the point spread function; same size as B.
%   center  [row, col] = indices of center of PSF.
%    alpha  Regularization parameter.
%             Default parameter chosen by generalized cross validation.
%
%  Output:
%        X  Array containing computed restoration.
%    alpha  Regularization parameter used to construct restoration.

% Reference: See Chapter 6, 
%            "Deblurring Images - Matrices, Spectra, and Filtering"
%            by P. C. Hansen, J. G. Nagy, and D. P. O'Leary,
%            SIAM, Philadelphia, 2006.

%
% Check number of inputs and set default parameters.
%
if (nargin < 3)
   error('B, PSF, and center must be given.')
end
if (nargin < 4)
   alpha = [];
end

%
% Use the FFT to compute the eigenvalues of the BCCB blurring matrix.
%
S = fft2( circshift(PSF, 1-center) );
s = S(:);

%
% If a regularization parameter is not given, use GCV to find one.
%
bhat = fft2(B);
bhat = bhat(:);
if (ischar(alpha) | isempty(alpha))
  alpha = gcv_tik(s, bhat);
end 
  
%
% Compute the Tikhonov regularized solution.
%
D = conj(s).*s + abs(alpha)^2;
bhat = conj(s) .* bhat;
xhat = bhat ./ D;
xhat = reshape(xhat, size(B));
X = real(ifft2(xhat));