function [X, tol] = tsvd_dct(B, PSF, center, tol)
%TSVD_DCT Truncated SVD image deblurring using the DCT algorithm.
%
%function [X, tol] = tsvd_dct(B, PSF, center, tol)
%
%           X = tsvd_dct(B, PSF, center);
%           X = tsvd_dct(B, PSF, center, tol);
%    [X, tol] = tsvd_dct(B, PSF, ...);
%
%  Compute restoration using a DCT-based truncated spectral factorization.
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
% Use the DCT to compute the eigenvalues of the symmetric 
% BTTB + BTHB + BHTB + BHHB blurring matrix.
%
e1 = zeros(size(PSF)); e1(1,1) = 1;
% Check to see if the built-in dct2 function is available; if not, 
% use our simple codes.
if exist('dct2') == 2
  bhat = dct2(B);
  S = dct2( dctshift(PSF, center) ) ./ dct2(e1);
else
  bhat = dcts2(B);
  S = dcts2( dctshift(PSF, center) ) ./ dcts2(e1);
end

%
% If a regularization parameter is not given, use GCV to find one.
%
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
% Check again to see if the built-in dct2 function is available.
if exist('dct2') == 2
  X = idct2(bhat .* Sfilt);
else
  X = idcts2(bhat .* Sfilt);
end