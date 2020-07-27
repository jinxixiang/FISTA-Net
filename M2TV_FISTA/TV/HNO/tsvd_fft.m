function [X, tol] = tsvd_fft(B, PSF, center, tol)
%TSVD_FFT Truncated SVD image deblurring using the FFT algorithm.
%
%function [X, tol] = tsvd_fft(B, PSF, center, tol)
%
%           X = tsvd_fft(B, PSF, center);
%           X = tsvd_fft(B, PSF, center, tol);
%    [X, tol] = tsvd_fft(B, PSF, ...);
%
%  Compute restoration using an FFT-based truncated spectral factorization.
%
%  Input:
%        B  Array containing blurred image.
%      PSF  Array containing the point spread function; same size as B.
%   center  [row, col] = indices of center of PSF.
%      tol  Regularization parameter (truncation tolerance).
%             Default parameter chosen by generalized cross validation.
%
%  Output:
%        X  Array containing computed restoration.
%      tol  Regularization parameter used to construct restoration.

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
   tol = [];
end

%
% Use the FFT to compute the eigenvalues of the BCCB blurring matrix.
%
S = fft2( circshift(PSF, 1-center) );

%
% If a regularization parameter is not given, use GCV to find one.
%
bhat = fft2(B);
if (ischar(tol) | isempty(tol))
  tol = gcv_tsvd(S(:), bhat(:));
end

%
% Compute the TSVD regularized solution.
%
Phi = (abs(S) >= tol);
idx = (Phi~=0);
Sfilt = zeros(size(Phi));
Sfilt(idx) = Phi(idx) ./ S(idx);
X = real(ifft2(bhat .* Sfilt));